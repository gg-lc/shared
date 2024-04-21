def chunk_expand(docs: List[Document], chunk_size: int = 2048) -> List[Document]:
    logging.info('start chunk expand')
    _start = time.time()
    # copy docs to avoid modifying the original list
    docs = copy.deepcopy(docs)
    for i, doc in tqdm(enumerate(docs)):
        expand_content = doc.page_content
        for j in range(i + 1, len(docs)):
            expanded = False
            # expand forward first
            if docs[j].metadata['source'] == doc.metadata['source']:
                if len(expand_content) + len(docs[j].page_content) > chunk_size:
                    break
                expand_content += docs[j].page_content
                expanded = True
            # expand backward
            if i - (j - i) >= 0 and docs[i - (j - i)].metadata['source'] == doc.metadata['source']:
                if len(expand_content) + len(docs[i - (j - i)].page_content) > chunk_size:
                    break
                expand_content = docs[i - (j - i)].page_content + expand_content
                expanded = True
            # if no expansion, break
            if not expanded:
                break
        # update metadata
        doc.metadata.update({'expand': expand_content})
    logging.info('chunk expand finished, time = {:.2f} sec'.format(time.time() - _start))
    return docs


def chunk_multi_index(docs: List[Document], model, key: bool = True, summary: bool = True,
                      use_expanded_content: bool = False) -> List[Document]:
    logging.info('start chunk multi index')
    logging.warning('temperature of the model should >> 0.0')
    _start = time.time()
    # copy docs to avoid modifying the original list
    multi_index_docs = []
    for doc in tqdm(docs):
        multi_index_docs.append(doc)
        for i in range(5):
            outputs = model.invoke(prompt_template.prompt_extract_keywords.format_messages(
                context=doc.page_content if not use_expanded_content else doc.metadata['expand']))
            lines = outputs.content.split('\n')
            keys, sums = None, None
            for line in lines:
                if line.startswith('keywords'):
                    keys = '@'.join(line.split('@')[1:])
                if line.startswith('summary'):
                    sums = '@'.join(line.split('@')[1:])
            if (key and keys is not None) and (summary and sums is not None):
                if key:
                    multi_index_docs.append(Document(page_content=keys, metadata=doc.metadata))
                if summary:
                    multi_index_docs.append(Document(page_content=sums, metadata=doc.metadata))
                break
            else:
                logging.warning(f'[RETRY] failed to extract keywords and summary from {outputs}.')
    logging.info('chunk multi index finished, time = {:.2f} sec'.format(time.time() - _start))
    return multi_index_docs
