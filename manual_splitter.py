import os
import logging
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

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
