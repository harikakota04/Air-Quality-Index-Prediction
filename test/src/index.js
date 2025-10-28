import React from "react";
import ReactDOM from "react-dom/client";
import App from "D:\test\src\app.tsx";
import "./index.css"; // Optional: If you have a CSS file for global styles

// Render the App component into the root div in index.html
const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);