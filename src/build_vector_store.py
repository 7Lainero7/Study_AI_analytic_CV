import json
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

DOCUMENTS_PATH = "./data/processed/documents.jsonl"
METADATA_PATH = "./data/processed/metadata.jsonl"
CHROMA_PATH = "./vectorstore/chroma_db"

def load_documents_and_metadata():
    documents = []
    metadatas = []
    ids = []

    with open(DOCUMENTS_PATH, "r", encoding="utf-8") as f_doc, \
         open(METADATA_PATH, "r", encoding="utf-8") as f_meta:

        for line_doc, line_meta in zip(f_doc, f_meta):
            doc = json.loads(line_doc)
            meta = json.loads(line_meta)

            doc_id = doc["id"]
            doc_text = doc["text"]

            # Пропускаем пустые документы
            if not doc_text or len(doc_text.strip()) < 50:
                continue

            ids.append(doc_id)
            documents.append(doc_text)
            
            # УЛУЧШЕННЫЕ МЕТАДАННЫЕ с фильтруемыми полями
            metadatas.append({
                "id": meta["id"],
                "url": meta["url"].strip(),
                "desired_position": meta["desired_position"].lower() if meta["desired_position"] else "",
                "location": meta["location"].lower() if meta["location"] else "",
                "total_experience_months": meta["total_experience_months"],
                "specialty_category": meta["specialty_category"].lower() if meta["specialty_category"] else "",
                "all_skills": ", ".join(meta["skills"]).lower() if meta["skills"] else "",
                "top_skills": ", ".join(meta["top_5_skills"]).lower() if meta.get("top_5_skills") else ""
            })

    print(f"✅ Загружено {len(documents)} документов.")
    print(f"📊 Пример метаданных: {metadatas[0] if metadatas else 'Нет данных'}")
    return ids, documents, metadatas

def main():
    os.makedirs(CHROMA_PATH, exist_ok=True)

    print("📥 Загрузка документов...")
    ids, documents, metadatas = load_documents_and_metadata()

    if not documents:
        print("❌ Нет документов для обработки!")
        return

    print("🧠 Генерация эмбеддингов...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(documents, show_progress_bar=True, batch_size=32).tolist()

    print("💾 Сохранение в ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(allow_reset=True))

    # Удаляем старую коллекцию (если есть)
    try:
        client.delete_collection("resumes")
    except:
        pass

    collection = client.create_collection(
        name="resumes",
        metadata={"hnsw:space": "cosine"},
        # Оптимизация для фильтрации
        embedding_function=None  # Используем предрасчитанные эмбеддинги
    )

    # Добавляем данные партиями для больших наборов
    batch_size = 1000
    for i in tqdm(range(0, len(ids), batch_size), desc="Добавление в ChromaDB"):
        batch_ids = ids[i:i+batch_size]
        batch_embeddings = embeddings[i:i+batch_size]
        batch_documents = documents[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]
        
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_documents,
            metadatas=batch_metadatas
        )

    print(f"✅ Векторное хранилище сохранено. Всего: {collection.count()} резюме.")
    
    # Проверка доступности коллекции
    print("🔍 Проверка коллекции...")
    test_query = "Python разработчик"
    test_results = collection.query(
        query_texts=[test_query],
        n_results=3,
        include=["documents", "metadatas"]
    )
    print(f"📋 Пример поиска по '{test_query}':")
    for i, (doc, meta) in enumerate(zip(test_results['documents'][0], test_results['metadatas'][0])):
        print(f"\n  {i+1}. {meta.get('desired_position', 'N/A').title()}")
        print(f"     Навыки: {meta.get('all_skills', 'N/A')[:100]}...")

if __name__ == "__main__":
    main()