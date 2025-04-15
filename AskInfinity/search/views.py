# search/views.py
from django.shortcuts import render, redirect
from django.http import JsonResponse
import os
import io
import base64
import tempfile
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_tavily import TavilySearch
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image

load_dotenv()
GOOGLE_API_KEY = '...............'           ##add you api key here
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
TAVILY_API_KEY = '.............'       ##add you api key here
os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)

def home(request):
    return render(request, 'home.html')  # This renders your app's main page

def ask_gemini(message, file_path=None, file_type=None, use_web=False):
    """Process user query with optional file context or web search"""
    
    # If there's a file, handle it based on type
    if file_path:
        if file_type == 'pdf':
            return process_pdf_query(message, file_path)
        elif file_type == 'image':
            return process_image_query(message, file_path)
    
    # If web search is requested, use Tavily
    if use_web:
        return process_web_query(message)
    
    # Standard text query processing
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest",
                                max_tokens=1024,
                                temperature=0.7)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful chatbot that answers the queries of users in helpful manner. Always answer in increasing order of complexity. Do not answer the question if you do not know the answer.",
            ),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm
    response = chain.invoke(
        {
            "input": message,
        }
    )
    return response.content

def process_web_query(message):
    """Process a query using Tavily web search for up-to-date information"""
    try:
        # Initialize Tavily search tool
        search_tool = TavilySearch(max_results=5,
                                topic="general")
        
        # Set up LLM
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", 
                                     temperature=0.7,
                                     max_tokens=1024)
        
        # Create RAG prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an AI assistant with access to real-time web search results.
                    Use the provided web search results to answer the user's question.
                    Always cite your sources at the end of your response with numbered links to the original sources.
                    If the search results don't contain relevant information, you can respond based on your knowledge,
                    but clearly indicate when you're doing so.
                    
                    Web search results:
                    {context}
                    """
                ),
                ("human", "{input}"),
            ]
        )
        
        # Execute search
        search_results = search_tool.invoke({'query':message})
        
        # Check if we got meaningful results
        if not search_results:
            # Fall back to standard response if no search results
            return ask_gemini(message)
        
        # Format results for context
        formatted_results = []
        for idx, result in enumerate(search_results):
            formatted_result = f"[{idx+1}] {result['title']}\nURL: {result['url']}\nContent: {result['content']}"
            formatted_results.append(formatted_result)
        
        context = "\n\n".join(formatted_results)
        
        # Generate response with search context
        response = llm.invoke(prompt.format(context=context, input=message))
        
        return response.content
    
    except Exception as e:
        # Handle any errors
        return f"I encountered an error searching the web: {str(e)}. Let me answer based on what I already know.\n\n{ask_gemini(message)}"

def process_pdf_query(message, pdf_path):
    """Process a query with context from a PDF file using RAG"""
    try:
        # Load and split the PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        # Create embeddings and vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_documents(splits, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        
        # Set up LLM
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", 
                                     temperature=0.7,
                                     max_tokens=1024)
        
        # Create RAG prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an AI assistant for answering questions based on specific documents.
                    Use the provided context to answer the user's question. 
                    If you cannot find the answer in the context, say so and don't make up information.
                    
                    Context:
                    {context}
                    """
                ),
                ("human", "{input}"),
            ]
        )
        
        # Create document chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        
        # Create retrieval chain
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        # Invoke chain
        response = retrieval_chain.invoke({"input": message})
        
        # Clean up temporary file after processing
        try:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
        except Exception as e:
            print(f"Error removing temporary file: {e}")
            
        return response["answer"]
    
    except Exception as e:
        # Handle any errors
        return f"I encountered an error processing your PDF: {str(e)}. Please try again or upload a different file."

def process_image_query(message, image_path):
    """Process a query about an image using Gemini's multimodal capabilities"""
    try:
        # Load the image
        img = Image.open(image_path)
        
        # Initialize Gemini Pro Vision model
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        
        # Process the image and query
        response = model.generate_content([
            "You are a helpful assistant that describes images and answers questions about them.",
            image_path if isinstance(image_path, Image.Image) else img,
            message
        ])
        
        # Clean up temporary file after processing
        try:
            if os.path.exists(image_path) and isinstance(image_path, str):
                os.remove(image_path)
        except Exception as e:
            print(f"Error removing temporary file: {e}")
            
        return response.text
    
    except Exception as e:
        # Handle any errors
        return f"I encountered an error processing your image: {str(e)}. Please try again or upload a different file."

def save_uploaded_file(uploaded_file):
    """Save an uploaded file to a temporary location and return the path"""
    # Create a temporary file with the same extension
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
    
    # Write the file content
    for chunk in uploaded_file.chunks():
        temp_file.write(chunk)
    temp_file.close()
    
    return temp_file.name

def get_file_type(file_name):
    """Determine file type based on extension"""
    extension = os.path.splitext(file_name)[1].lower()
    
    if extension == '.pdf':
        return 'pdf'
    elif extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
        return 'image'
    else:
        return 'unknown'

# Create your views here.
def chatbot(request):
    if request.method == 'POST':
        message = request.POST.get('message', '')
        use_web = request.POST.get('use_web', 'false').lower() == 'true'
        
        # Check if a file was uploaded
        uploaded_file = request.FILES.get('file')
        file_path = None
        file_type = None
        
        if uploaded_file:
            # Save the file and get its path
            file_path = save_uploaded_file(uploaded_file)
            file_type = get_file_type(uploaded_file.name)
        
        # Process the message with file context or web search
        response = ask_gemini(message, file_path, file_type, use_web)
        
        return JsonResponse({'message': message, 'response': response})

    return render(request, 'chatbot.html')