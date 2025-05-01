from langchain.chains import RetrievalQA
import logging

persist_directory = ".chroma"

def ask(query, llm, vectordb):
    # embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    # vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    retriever = vectordb.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=True
    )
    logging.info(f"LLM type: {type(llm)}, model: {getattr(llm, 'model_name', 'n/a')}")
    result = qa.invoke({"query": query})

    answer = result.get("result", "No answer found.")
    # source_docs = result.get("source_documents", [])
    # print(source_docs)

    
    # if source_docs:
    #     print("\nSources:")
    #     for i, doc in enumerate(source_docs):
    #         print(f"[{i+1}] {doc.metadata.get('source', 'Unknown source')}")
    # else:
    #     print("\nNo sources found.")

    return answer
