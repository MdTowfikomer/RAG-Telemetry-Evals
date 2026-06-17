import asyncio
import os
from datetime import datetime, UTC
from pathlib import Path
from typing import List
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from sqlmodel import Session, col, select

from backend.api.dependencies import factory, get_db
from backend.core.models import UploadedFile
from backend.ingest import ingest

router = APIRouter(prefix="/documents", tags=["documents"])

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data"

ingestion_lock = asyncio.Lock()


async def run_ingestion(file_ids: List[UUID]):
    async with ingestion_lock:
        engine = factory.get_engine()
        with Session(engine) as session:
            for fid in file_ids:
                db_file = session.get(UploadedFile, fid)
                if db_file:
                    db_file.status = "processing"
                    db_file.updated_at = datetime.now(UTC)
                    session.add(db_file)
            session.commit()

        try:
            # Run the synchronous ingest script in a separate thread
            await asyncio.to_thread(ingest, factory)

            with Session(engine) as session:
                for fid in file_ids:
                    db_file = session.get(UploadedFile, fid)
                    if db_file:
                        db_file.status = "indexed"
                        db_file.updated_at = datetime.now(UTC)
                        session.add(db_file)
                session.commit()
        except Exception as e:
            print(f"Error during document ingestion: {e}")
            with Session(engine) as session:
                for fid in file_ids:
                    db_file = session.get(UploadedFile, fid)
                    if db_file:
                        db_file.status = "failed"
                        db_file.error_message = str(e)
                        db_file.updated_at = datetime.now(UTC)
                        session.add(db_file)
                session.commit()


@router.get("", response_model=List[UploadedFile])
async def get_documents(db: Session = Depends(get_db)):
    statement = select(UploadedFile).order_by(col(UploadedFile.created_at).desc())
    results = db.exec(statement).all()
    return results


@router.post("/upload", response_model=List[UploadedFile])
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    try:
        DATA_PATH.mkdir(parents=True, exist_ok=True)

        db_files = []
        file_ids = []

        for file in files:
            if not file.filename:
                continue

            ext = Path(file.filename).suffix.lower()
            if ext not in [".pdf", ".txt", ".md"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file extension {ext}. Only .pdf, .txt, and .md are supported.",
                )

            fid = uuid4()
            safe_filename = f"{fid}_{file.filename}"
            dest_path = DATA_PATH / safe_filename

            try:
                content = await file.read()
                file_size = len(content)
                with open(dest_path, "wb") as buffer:
                    buffer.write(content)
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to save file {file.filename}: {str(e)}",
                )

            db_file = UploadedFile(
                id=fid,
                filename=file.filename,
                file_size=file_size,
                status="pending",
            )
            db.add(db_file)
            db_files.append(db_file)
            file_ids.append(fid)

        db.commit()
        for db_file in db_files:
            db.refresh(db_file)

        background_tasks.add_task(run_ingestion, file_ids)

        return db_files
    except HTTPException:
        raise
    except Exception as error:
        print(f"Error in upload endpoint: {error}")
        raise HTTPException(status_code=500, detail=str(error))


@router.delete("/{document_id}")
async def delete_document(
    document_id: UUID,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    db_file = db.get(UploadedFile, document_id)
    if not db_file:
        raise HTTPException(status_code=404, detail="Document not found")

    safe_filename = f"{db_file.id}_{db_file.filename}"
    file_path = DATA_PATH / safe_filename
    if file_path.exists():
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Failed to delete file from disk: {e}")

    db.delete(db_file)
    db.commit()

    # Trigger background ingestion to rebuild the index without the deleted file
    background_tasks.add_task(run_ingestion, [])

    return {"message": "Document deleted and index rebuild triggered"}
