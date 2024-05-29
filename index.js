import express from "express";
import bodyParser from "body-parser";
import cors from "cors";
import { askSalesAgent } from "./query_data.js";
const app = express();
app.use(cors());
app.use(bodyParser.json());
app.use(express.static("public"));
app.use(express.urlencoded({ extended: true }));

app.get("/", (req, res) => {
  res.sendFile(__dirname + "/public/index.html");
});

app.post("/chat", async (req, res) => {
  try {
    const { userInput } = req.body;
    console.log("Form User Input:", userInput);

    const response = await askSalesAgent(userInput);

    console.log("Form Response:", response);
    res.json({ response: response.text });
  } catch (error) {
    console.error(error);
    res.status(500).send("An error occurred");
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
