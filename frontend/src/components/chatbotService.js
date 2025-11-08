// chatbotService.js

// Lấy API URL từ environment variable hoặc sử dụng default
// Trong production: VITE_API_URL=https://legalbizai.hypersona.vn/api
// Trong development: VITE_API_URL=http://localhost:1142
// const API_URL = import.meta.env.VITE_API_URL || 'https://legalbizai.hypersona.vn/api';

export const sendMessageChatService = async (promptInput, model) => {
    // Sử dụng API_URL từ environment variable
    const apiEndpoint = `https://legalbizai.hypersona.vn/api/stream`;
    
    const response = await fetch(apiEndpoint, {
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