import os
import json
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Пути
CHROMA_PATH = "./vectorstore/chroma_db"

def test_query():
    print("🔍 Загрузка модели эмбеддингов...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("📂 Подключение к ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(allow_reset=False))
    collection = client.get_collection("resumes")

    # Пример запроса
    query_text = "Найди Vue-разработчиков из Москвы с опытом SEO и адаптивной верстки"
    print(f"\n💬 Запрос: {query_text}")

    # Эмбеддинг запроса
    query_embedding = model.encode(query_text).tolist()

    # Фильтрация: только Москва + опыт >= 3 лет (36 месяцев)
    where_filter = {
        "$and": [
            {"location": {"$eq": "Москва"}},
            {"total_experience_months": {"$gte": 36}}
        ]
    }

    print("\n🔎 Выполняем поиск с фильтрацией (Москва, опыт ≥ 3 лет)...")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )

    # Вывод результатов
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not ids:
        print("❌ Ничего не найдено по заданным условиям.")
        return

    print(f"\n✅ Найдено {len(ids)} резюме:\n")
    for i, (res_id, doc, meta, dist) in enumerate(zip(ids, documents, metadatas, distances), 1):
        print(f"--- Результат {i} ---")
        print(f"🆔 ID: {res_id}")
        print(f"🌐 URL: {meta.get('url', '—')}")
        print(f"📍 Локация: {meta.get('location', '—')}")
        print(f"💼 Должность: {meta.get('desired_position', '—')}")
        print(f"📅 Опыт (мес): {meta.get('total_experience_months', '—')}")
        print(f"🔧 Навыки (top): {meta.get('top_skills', '—')}")
        print(f"📊 Релевантность (расстояние): {dist:.3f}")
        # Показываем первые 200 символов описания
        snippet = doc.split("[Описание: ")[-1][:200] + "..." if "[Описание: " in doc else doc[:200] + "..."
        print(f"📄 Фрагмент: {snippet}")
        print()

if __name__ == "__main__":
    test_query()