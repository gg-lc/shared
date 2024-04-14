import json
import logging
import os
import re
import warnings
from typing import List

import torch
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] (%(module)s#%(lineno)d): %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO)
warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core.vectorstores")


def load_split_gpt3(file_root: str, gpt_save_path: str = '/root/.cache/rag/split', re_split=False) -> List[Document]:
    """
    load and split a list of documents into chunks and save them to the given path (txt files, separated by '\n\n')
    :param file_root:
    :param gpt_save_path:
    :param re_split: maximum len(chunk)
    :return: List[Document]
    """
    def load_files(pdf_root: str) -> List[Document]:
        """load all pdf files in the given root directory into single page documents"""
        # use PyPDFLoader to load pdf files as Documents(page_content, metadata{source, page}})
        loaders = [PyPDFLoader(os.path.join(pdf_root, _)) for _ in os.listdir(pdf_root)]
        # load all pages in each pdf file into a single page document
        docs = [Document(
            page_content=' '.join([_.page_content for _ in loader.load()]),
            metadata={'source': loader.load()[0].metadata['source'].split('/')[-1][:-4]}
        ) for loader in loaders]
        return docs

    def split_file_gpt(doc: Document, save_path=gpt_save_path):
        logging.info(f"[SPLIT] split and save {doc.metadata['source']} start.")
        split_prompt = """
        【要求】请帮我切分下面的政策文档，使其更适合基于RAG技术的信息检索和问答处理。操作需遵循以下步骤：
        1. 删除文档中所有不必要的空格和换行符。
        2. 根据文档内容的逻辑内容，将其切分成独立的适当大小的文档块。每个块之间使用两个换行符隔开以明确区分。
        3. 确保每个文档块的内容基于逻辑内容的自然分界线（如主题变化、逻辑内容）进行切分。
        4. 【重要】不要将连续的多条规则、流程、步骤、细则切分成多个块！
        5. 【重要】每个文档块应保持信息的完整性和独立性，即使这意味着需要将关联紧密的规则或段落合并在一起。
        6. 【重要】保留文档中的全部内容，不进行删除或修改，只做结构上的调整
        7. 【重要】每个文档块不要太长以避免影响检索精确性；也不要太短，将连续的、较短的段落（1-2句话）合并到一个块中。
        8. 完成后，只输出根据上述要求切分整理好的文档块，不要附加任何其他说明或信息，不要添加任何前缀和后缀！
        \n\n\n{text}
        """
        extract_prompt = """
        请阅读下面这个天津大学的政策制度文档，分析文档的元数据信息，包括：文档标题、发布时间(年月)，并按以下格式输出：
            metadata:文档标题；发布于:发布时间
        例如:
            metadata:天津大学学生手册；发布于:2021年9月
            metadata:天津大学智能与计算学部2019年研究生国家奖学金评审方案；发布于:2023年1月
        请严格按照要求的格式输出，不要包含任何其他信息。
        \n\n以下是政策制度文档：\n\n{text}
        """
        openai_base = 'https://openkey.cloud/v1'
        openai_key = 'sk-Bu13cA5CyKlNaqZk3dC51bF99c2a44899973Fc9a723b0c8b'
        openai_model = 'gpt-3.5-turbo'  # NOTE: 0.05-0.1￥/doc
        openai_chat = ChatOpenAI(
            model=openai_model,
            openai_api_base=openai_base,
            openai_api_key=openai_key,
            temperature=0.5,
        )
        max_length, cnt = 12000, 1  # 15k chars
        metadata_output = openai_chat.invoke(extract_prompt.format(text=doc.page_content[:max_length])).content

        split_output = ''
        while max_length * (cnt - 1) < len(doc.page_content) - 50:
            # print('*'*50 + '\n' + split_prompt.format(text=doc.page_content[max_length*(cnt-1):max_length*cnt]))
            split_output += openai_chat.invoke(split_prompt.format(
                text=doc.page_content[min(0, max_length * (cnt - 1) - 100):max_length * cnt])).content
            cnt += 1

        # save into save_path as txt
        with open(os.path.join(save_path, f'{doc.metadata["source"]}.txt'), 'w') as f:
            f.write(metadata_output + '\n\n')
            f.write(split_output)
        logging.info(f"split and save {doc.metadata['source']} done.")

    # check save_path: if not empty -> press 1 to clear and re-split; press other to skip
    if os.path.isdir(gpt_save_path):
        if re_split:
            logging.warning('remove all files in {}!'.format(gpt_save_path))
            os.system(f'rm -rf {gpt_save_path}')
    if not os.path.isdir(gpt_save_path):
        logging.info('start split and save...')
        os.makedirs(gpt_save_path)
        import threading
        threads = []
        for doc in load_files(file_root):
            t = threading.Thread(target=split_file_gpt, args=(doc,))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

    # load split files as Documents
    split_files = os.listdir(gpt_save_path)
    post_spliter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=0)
    docs = []
    for file in split_files:
        with open(os.path.join(gpt_save_path, file), 'r') as f:
            contents = f.read().split('\n\n')
            metadata = contents[0]
            contents = ''.join(contents[1:])
            for chunk in post_spliter.split_text(contents):
                docs.append(Document(
                    page_content=metadata + '\n' + chunk,
                    metadata={'source': file[:-4]}
                ))
        logging.info(f'read {file} done.')
    return docs


def load_split_manual(file_root: str, chunk_size=512,
                      save_path: str = '/root/.cache/rag/split_manual', re_split=False) -> List[Document]:
    # fixme: save Docs into save_path
    logging.info('start loading manual split docs...')
    spliter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_size // 3)
    docs = []
    for file in os.listdir(file_root):
        source = file.split('；')[0]
        metadata = ''.join(file.split('.')[:-1])
        contexts = open(os.path.join(file_root, file), 'r').read().split('\n\n')
        i = 0
        while i < len(contexts):
            context = contexts[i]
            # small chunks
            if len(context) < chunk_size:
                while len(context) < chunk_size and i + 1 < len(contexts):
                    if len(context + contexts[i + 1]) > chunk_size:
                        break
                    context += contexts[i + 1]
                    i += 1
                context = context.replace('\n', '')
                if len(context):  # skip '\n' chunks
                    docs.append(Document(page_content=context, metadata={'source': source, 'metadata': metadata}))
            # large chunks
            else:
                context = context.replace('\n', '')
                for chunk in spliter.split_text(context):
                    docs.append(Document(page_content=chunk, metadata={'source': source, 'metadata': metadata}))
            i += 1
    logging.info('loaded {} manual split docs'.format(len(docs)))
    return docs
