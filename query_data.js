import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { CSVLoader } from "langchain/document_loaders/fs/csv";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { DocxLoader } from "langchain/document_loaders/fs/docx";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { OpenAI } from "@langchain/openai";
import { RetrievalQAChain } from "langchain/chains";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { OpenAIEmbeddings } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { ChatMessageHistory } from "langchain/stores/message/in_memory";

import dotenv from "dotenv";
import fs from "fs";
dotenv.config();

// Initialize the document loader with supported file formats
const loader = new DirectoryLoader("./data", {
  ".json": (path) => new JSONLoader(path),
  ".txt": (path) => new TextLoader(path),
  ".csv": (path) => new CSVLoader(path),
  ".pdf": (path) => new PDFLoader(path),
  ".docx": (path) => new DocxLoader(path),
});

//Load documents from the specified directory
console.log("Loading docs...");
const docs = await loader.load();
console.log("Docs loaded.");

const VECTOR_STORE_PATH = "Data.index";

function normalizeDocuments(docs) {
  return docs.map((doc) => {
    if (typeof doc.pageContent === "string") {
      return doc.pageContent;
    } else if (Array.isArray(doc.pageContent)) {
      return doc.pageContent.join("\n");
    }
  });
}

const salesAgentPrompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    `You are a virtual sales representative for an online store. Your primary goal is to assist customers with their inquiries, provide detailed information about products, help them find what they're looking for, and facilitate the purchasing process. Here are some key guidelines to follow:

    Friendly and Professional Tone: Always maintain a friendly and professional tone. Greet customers warmly and be courteous throughout the conversation.
    Product Knowledge: Be knowledgeable about all the products listed in the store. Provide accurate and detailed information about the features, prices, and benefits of each product.
    Customer Assistance: Help customers find products based on their needs and preferences. Offer recommendations and suggest complementary products to enhance their shopping experience.
    Handling Queries: Respond promptly to customer queries. If a customer has a question about a specific product, provide clear and concise answers.
    Facilitate Purchases: Guide customers through the purchasing process. Assist them with adding items to their cart, checking out, and completing their orders.
    Problem Resolution: Address any issues or concerns the customers might have. If a problem cannot be resolved immediately, assure the customer that you will escalate it to the appropriate team.
    Upselling and Cross-selling: Where appropriate, suggest additional products that complement the customer's purchase to increase the value of their order.
    Personalization: Personalize interactions by using the customer's name if provided and referencing their past interactions or preferences.
    Example Interactions
    Greeting:
    "Hello! Welcome to our store. How can I assist you today?"
    
    Product Inquiry:
    "Sure, the Laptop Pro 15 is a high-performance device featuring an Intel i7 processor, 16GB RAM, and a 512GB SSD. It’s perfect for both professional and personal use. Would you like to know more or add it to your cart?"
    
    Recommendation:
    "I see you're interested in the Bluetooth Headphones. We also have a great offer on a portable Bluetooth speaker that pairs perfectly with those headphones. Would you like to check it out?"
    
    Assistance with Purchase:
    "To complete your purchase, please add the items to your cart and proceed to checkout. If you need any help during the process, feel free to ask!"
    
    Problem Resolution:
    "I'm sorry to hear you're having an issue with your order. Let me look into that for you right away. Could you please provide your order number?"
    
    By adhering to these guidelines, you will ensure a smooth and satisfying shopping experience for all customers.
    Always answer the user's questions based on the below context:
    {context}`,
  ],
  ["human", "{question}"],
]);

const messageHistory = new ChatMessageHistory();

export const askSalesAgent = async (question) => {
  const model = new OpenAI({
    temperature: 1,
    maxTokens: 300,
    modelName: "gpt-3.5-turbo-1106",
  });

  let vectorStore;

  console.log("Checking for existing vector store...");
  if (fs.existsSync(VECTOR_STORE_PATH)) {
    console.log("Loading existing vector store...");
    vectorStore = await HNSWLib.load(VECTOR_STORE_PATH, new OpenAIEmbeddings());
    console.log("Vector store loaded.");
  } else {
    console.log("Creating new vector store...");
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
    });
    const normalizedDocs = normalizeDocuments(docs);
    const splitDocs = await textSplitter.createDocuments(normalizedDocs);

    vectorStore = await HNSWLib.fromDocuments(
      splitDocs,
      new OpenAIEmbeddings()
    );

    await vectorStore.save(VECTOR_STORE_PATH);

    console.log("Vector store created.");
  }
  await messageHistory.addMessage({
    content: question,
    additional_kwargs: {},
  });

  console.log("Creating retrieval chain...");
  const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever(), {
    prompt: salesAgentPrompt,
    messageHistory: messageHistory,
  });

  console.log("Querying chain...");
  const response = await chain.invoke({ query: question });
  console.log({ response });
  return response;
};

// const question = "i am looking for Laptop Pro 15 whats its price?";

// await askSalesAgent(question);
