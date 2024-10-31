import pickle
import pandas as pd
import os
from loguru import logger
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter #用excel就不需要文本切割，如果后续加了文本的rag需要用到
from langchain_core.documents.base import Document
from FlagEmbedding import FlagReranker
from src.config.config import (
    embedding_path,
    embedding_model_name,
    doc_dir, qa_dir, xlsx_dir,
    knowledge_pkl_path,
    data_dir,
    vector_db_dir,
    rerank_path,
    rerank_model_name,
    chunk_size,
    chunk_overlap,
    select_num,
    retrieval_num
)
# from src.config.config import (
#     embedding_path,
#     embedding_model_name,
#     doc_dir, qa_dir, xlsx_dir,
#     knowledge_pkl_path,
#     data_dir,
#     vector_db_dir,
#     rerank_path,
#     rerank_model_name,
#     chunk_size,
#     chunk_overlap,
#     select_num,
#     retrieval_num
# )


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Data_process_for_server(metaclass=Singleton):
    #auto run model loading...
    def __init__(self):
        self.embedding_model = None
        self.rerank_model = None
        self.vector_db = None
        self.qa_dict = None
        
        # Load all models and data on initialization
        self.load_embedding_model()
        self.load_rerank_model()
        self.read_excel_to_dict()
        self.load_vector_db()
        
        logger.info("Data_process_for_server initialized with all models and data loaded.")

    def load_embedding_model(self, model_name=embedding_model_name, device='cpu', normalize_embeddings=True):
        if self.embedding_model is None:
            if not os.path.exists(embedding_path):
                os.makedirs(embedding_path, exist_ok=True)
            embedding_model_path = os.path.join(embedding_path, model_name.split('/')[1] + '.pkl')
            logger.info('Loading embedding model...')
            if os.path.exists(embedding_model_path):
                try:
                    with open(embedding_model_path, 'rb') as f:
                        self.embedding_model = pickle.load(f)
                        logger.info('Embedding model loaded from file.')
                except Exception as e:
                    logger.error(f'Failed to load embedding model from {embedding_model_path}')
            if self.embedding_model is None:
                try:
                    self.embedding_model = HuggingFaceBgeEmbeddings(
                        model_name=model_name,
                        model_kwargs={'device': device},
                        encode_kwargs={'normalize_embeddings': normalize_embeddings})
                    logger.info('Embedding model loaded.')
                    with open(embedding_model_path, 'wb') as file:
                        pickle.dump(self.embedding_model, file)
                except Exception as e:
                    logger.error(f'Failed to load embedding model: {e}')
        return self.embedding_model

    def load_rerank_model(self, model_name=rerank_model_name):
        if self.rerank_model is None:
            if not os.path.exists(rerank_path):
                os.makedirs(rerank_path, exist_ok=True)
            rerank_model_path = os.path.join(rerank_path, model_name.split('/')[1] + '.pkl')
            logger.info('Loading rerank model...')
            if os.path.exists(rerank_model_path):
                try:
                    with open(rerank_model_path, 'rb') as f:
                        self.rerank_model = pickle.load(f)
                        logger.info('Rerank model loaded from file.')
                except Exception as e:
                    logger.error(f'Failed to load rerank model from {rerank_model_path}')
            if self.rerank_model is None:
                try:
                    self.rerank_model = FlagReranker(model_name, use_fp16=True)
                    logger.info('Rerank model loaded.')
                    with open(rerank_model_path, 'wb') as file:
                        pickle.dump(self.rerank_model, file)
                except Exception as e:
                    logger.error(f'Failed to load rerank model: {e}')
                    raise
        return self.rerank_model

    def read_excel_to_dict(self):
        if self.qa_dict is None:
            file_path = os.path.join(xlsx_dir, 'character.xlsx')
            try:
                df = pd.read_excel(file_path, sheet_name='Sheet1')
                self.qa_dict = {row['用户提问']: row['AI回答'] for _, row in df.iterrows()}
                logger.info("Excel file read successfully.")
            except Exception as e:
                logger.error(f"Failed to read Excel file: {e}")
                raise
        return self.qa_dict

    def save_question_to_list(self):
        qa_dict = self.read_excel_to_dict()
        try:
            question_texts = list(qa_dict.keys())
            logger.info(f"Extracted {len(question_texts)} questions from the dictionary.")
            return question_texts
        except Exception as e:
            logger.error(f"Error in extracting: {e}")
            return None

    def create_vector_db(self):
        logger.info(f'Creating index...')
        emb_model = self.load_embedding_model()
        question_texts = self.save_question_to_list()
        documents = [Document(page_content=text) for text in question_texts]
        if question_texts is not None:
            self.vector_db = FAISS.from_documents(documents, emb_model)
            try:
                self.vector_db.save_local(vector_db_dir)
            except Exception as e:
                logger.error(f"Failed to save vector database: {e}")
        return self.vector_db

    def load_vector_db(self):
        if self.vector_db is None:
            emb_model = self.load_embedding_model()
            if not os.path.exists(vector_db_dir) or not os.listdir(vector_db_dir):
                self.vector_db = self.create_vector_db()
            else:
                self.vector_db = FAISS.load_local(vector_db_dir, emb_model, allow_dangerous_deserialization=True)
        return self.vector_db

    def retrieve(self, query, k=10):
        vector_db = self.load_vector_db()
        logger.info(f'Retrieving top {k} documents for query: {query}')
        retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": k})
        docs = retriever.invoke(query)
        return docs

    def rerank(self, query, docs, retrieval_num):
        reranker = self.load_rerank_model()
        doc_texts = [doc.page_content.strip() for doc in docs]
        logger.info(f'Running rerank for query: {query}')

        try:
            doc_score = {}
            for doc_text in doc_texts:
                score_pair = [query, doc_text]
                score = reranker.compute_score(score_pair)
                doc_score[doc_text] = score[0]

            sorted_pair = sorted(doc_score.items(), key=lambda item: abs(item[1]))
            top_n_pairs = sorted_pair[:retrieval_num]

            sorted_docs = [text for text, _ in top_n_pairs]
            sorted_scores = [score for _, score in top_n_pairs]

            logger.info(f'Document sorted: {sorted_docs}, score sorted: {sorted_scores}')
            return sorted_docs, sorted_scores

        except Exception as e:
            logger.error(f'Error during reranking: {e}')
            return [], []

    def return_answer(self, query, docs, retrieval_num):
        sorted_docs, sorted_scores = self.rerank(query, docs, retrieval_num)
        matched_answers = []
        qa_dict = self.read_excel_to_dict()

        for doc, score in zip(sorted_docs, sorted_scores):
            if abs(score) < 2:
                if doc in qa_dict:
                    matched_answers.append(qa_dict[doc])
                else:
                    matched_answers.append('')

        if not matched_answers:
            matched_answers.append('')

        for answer in matched_answers:
            print(answer)

        return matched_answers


# Initialize the singleton instance
data_processor = Data_process_for_server()

if __name__ == "__main__":
    dp = data_processor
    query = "心爱是一个什么样的人"
    docs = dp.retrieve(query, k=10)
    dp.return_answer(query, docs, retrieval_num)