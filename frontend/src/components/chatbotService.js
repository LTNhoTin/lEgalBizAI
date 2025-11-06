// chatbotService.js
export const sendMessageChatService = async (promptInput, model) => {
    // Use relative path /api/stream which will be proxied by nginx to backend
    const response = await fetch('/api/stream', {
      method: "post",
      body: JSON.stringify({
        message: promptInput,
        model: model
      }),
      headers: new Headers({
        "ngrok-skip-browser-warning": "69420",
        "Content-Type": "application/json"
      }),
    });
    
    if (!response.ok) {
      throw new Error("Network response was not ok");
    }
    
    const result = await response.json();
    return result;
  };