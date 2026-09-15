import React from "react";
import ReactDOM from "react-dom/client";
import "./index.css";
import Experiment from "./components/Experiment";

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
    <React.StrictMode>
        <Experiment />
    </React.StrictMode>
);
