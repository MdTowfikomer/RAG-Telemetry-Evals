# Frontend Implementation Plan (Issue #4)

This plan breaks down the development of the streaming React UI into 5 distinct, manageable phases.

### Phase 1: Project Setup & Core Layout
*   **Goal:** Establish the foundation of the React app.
*   **Tasks:**
    1.  Install and configure TailwindCSS for Vite.
    2.  Install `lucide-react` for icons.
    3.  Create the main 3-pane responsive layout in `App.tsx` using CSS Grid or Flexbox.
    4.  Create placeholder components for `LeftPane`, `ChatPane`, and `RightPane` and place them in the layout.

### Phase 2: Static Component Implementation
*   **Goal:** Build all UI components with static, hard-coded data.
*   **Tasks:**
    1.  Implement the `ChatPane`, including the input form and a few hard-coded `Message` components.
    2.  Implement the `RightPane` with several hard-coded `ContextCard` components.
    3.  Implement the `LeftPane` with the `Settings` and `History` components, showing placeholder controls and conversation items.

### Phase 3: State Management & Basic API Communication
*   **Goal:** Wire up the UI to the backend with non-streaming communication.
*   **Tasks:**
    1.  In `App.tsx`, create state using `useState` for `messages`, `contextDocs`, and `isLoading`.
    2.  Implement the `handleSendQuery` function.
    3.  On form submit, call `handleSendQuery` to make a `POST` request to the `/chat` (non-streaming) endpoint.
    4.  Use the response to update the `messages` and `contextDocs` state, verifying that the UI re-renders correctly.

### Phase 4: Implement Streaming Response
*   **Goal:** Integrate the Server-Sent Events (SSE) logic.
*   **Tasks:**
    1.  Modify `handleSendQuery` to first fetch context, then open an `EventSource` connection to `/chat/stream`.
    2.  Add an `onmessage` listener to the `EventSource`.
    3.  As token events arrive, append them to the last message in the `messages` state array.
    4.  Handle the `[DONE]` message to close the `EventSource` connection.

### Phase 5: Final Polish & Error Handling
*   **Goal:** Add final touches to make the UI robust and user-friendly.
*   **Tasks:**
    1.  Add a loading indicator that appears while the backend is processing the request.
    2.  Display a clear error message in the UI if the API call fails.
    3.  Implement the "Clear Chat" button functionality.
    4.  Add `lucide-react` icons for Send, Clear, etc., and perform any final styling adjustments.
