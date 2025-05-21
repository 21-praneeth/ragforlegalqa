
def rag_pipeline(question: str):
    import warnings
    warnings.filterwarnings("ignore")

    #!/usr/bin/env python
    # coding: utf-8
    
    # In[ ]:
    
    
    get_ipython().system('pip install langchain-community')
    get_ipython().system('pip install sentence_transformers')
    get_ipython().system('pip install faiss-cpu')
    get_ipython().system('pip install -U transformers')
    get_ipython().system('pip install accelerate')
    
    
    # In[ ]:
    
    
    import pandas as pd
    df = pd.read_csv('/content/constitution.csv')
    df.head()
    
    
    # In[ ]:
    
    
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    import pandas as pd
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=128,
        # length_function=len
    )
    
    chunked_data = []
    
    for _, row in df.iterrows():
        article_id = row["article_id"]
        article_text = str(row["article_desc"])
    
        article_intro = f"{article_id}: "
        full_text = article_intro + article_text
    
        chunks = text_splitter.split_text(full_text)
    
        for i, chunk in enumerate(chunks):
            chunked_data.append({
                "article_id": article_id,
                "chunk_id": i + 1,
                "chunk_text": chunk
            })
    
    chunked_df = pd.DataFrame(chunked_data)
    print(len(chunked_df))
    chunked_df.head()
    
    
    # In[ ]:
    
    
    import re
    from langchain.docstore.document import Document
    
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    documents = [Document(
        page_content=preprocess_text(row["chunk_text"]),
        metadata={"article_id": row["article_id"], "chunk_id": row["chunk_id"]}) for i, row in chunked_df.iterrows()]
    print(documents)
    
    
    # In[ ]:
    
    
    from langchain.embeddings import HuggingFaceEmbeddings
    import pandas as pd
    
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    texts = [doc.page_content for doc in documents]
    embeddings = embedding_model.embed_documents(texts)
    embed_df = pd.DataFrame({
        "text": texts,
        "embedding": embeddings
    
    })
    embed_df.head()
    # embeddings.shape[1]
    
    
    # In[ ]:
    
    
    import os
    os.environ["HF_TOKEN"]="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"
    
    
    # In[ ]:
    
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import numpy as np
    import faiss
    import torch
    
    model_name = "Qwen/Qwen3-4B"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    embeddings_np = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings_np)
    index = faiss.IndexFlatIP(embeddings_np.shape[1])
    index.add(embeddings_np)
    
    while True:
        user_query = input("\nAsk your legal question (or type 'exit'): ")
        if user_query.lower() == "exit":
            break
    
        cleaned_query = preprocess_text(user_query)
        query_embedding = embedding_model.embed_query(cleaned_query)
        query_embedding_np = np.array([query_embedding]).astype("float32")
        faiss.normalize_L2(query_embedding_np)
    
        D, I = index.search(query_embedding_np, k=5)
    
        relevant_chunks = []
        for score, idx in zip(D[0], I[0]):
            if score >= 0.4:
                doc = documents[idx]
                relevant_chunks.append(doc.page_content)
    
        if not relevant_chunks:
            print("\nNo, this does not come under legal queries.")
            continue
    
        context = "\n".join(relevant_chunks)
        system_prompt = (
        "You are a legal assistant. Answer the user's question based on the following context from the Constitution of India. "
        "Do not add any introductory or filler phrases. Respond only with the direct legal answer, in a formal and concise manner.\n\n"
        f"Context:\n{context}"
      )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
    
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
    
        model_inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.4,
            repetition_penalty=1.2,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id
        )
    
        generated_text = tokenizer.decode(output_ids[0][model_inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
        print("\nLLM's Answer:\n", generated_text.strip())
    

    # Customize your return value as needed
    return {{
        "answer": response,
        "summary": "Summary placeholder",
        "points": ["Point 1", "Point 2"]
    }}
