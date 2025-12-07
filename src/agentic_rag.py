# agentic_rag.py
import re
import json
import asyncio
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from gigachat import GigaChat

class AgenticRAGHandler:
    """Агент для интеллектуального поиска резюме с итеративным уточнением."""
    
    def __init__(self, model: SentenceTransformer, collection, giga_chat):
        self.model = model
        self.collection = collection
        self.giga_chat = giga_chat
        self.max_retries = 3
        
    async def _call_llm_with_retry(self, prompt: str, system_prompt: str = None) -> str:
        """Вызов LLM с повторными попытками."""
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        for attempt in range(self.max_retries):
            try:
                response = self.giga_chat.chat(full_prompt)
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"⚠️ Ошибка LLM, попытка {attempt+1}/{self.max_retries}: {e}")
                    await asyncio.sleep(1)
                else:
                    raise Exception(f"❌ Все попытки вызова LLM провалились: {e}")
    
    def _parse_agent_response(self, response: str) -> dict:
        """Парсинг структурированного ответа агента с обработкой невалидного JSON."""
        import re
        import json
        try:
            # 1. Удаляем markdown код
            cleaned = re.sub(r'```json\n?|\n?```', '', response)
            
            # 2. Удаляем начальные/конечные пробелы и кавычки
            cleaned = cleaned.strip()
            if cleaned.startswith('"') and cleaned.endswith('"'):
                cleaned = cleaned[1:-1]
            
            # 3. Экранируем проблемные символы в строковых значениях
            # Находим все строковые значения и экранируем в них переводы строк
            import json
            def fix_json_strings(match):
                text = match.group(1)
                # Экранируем специальные символы
                text = text.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                text = text.replace('"', '\\"')
                return f'"{text}"'
            
            # Регулярка для нахождения строковых значений
            pattern = r'"([^"\\]*(?:\\.[^"\\]*)*)"'
            fixed_json = re.sub(pattern, fix_json_strings, cleaned)
            
            # 4. Пробуем распарсить исправленный JSON
            try:
                return json.loads(fixed_json)
            except json.JSONDecodeError as e:
                # Если не удалось, пробуем более агрессивное исправление
                print(f"⚠️ Первая попытка парсинга не удалась: {e}")
                
                # Удаляем все непечатаемые символы кроме пробелов
                import re
                cleaned_final = re.sub(r'[\x00-\x1F\x7F]', ' ', cleaned)
                
                # Пробуем найти JSON между фигурными скобками
                json_match = re.search(r'\{.*\}', cleaned_final, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        pass
                
                # Возвращаем дефолтную структуру
                return {
                    "thought_process": "Не удалось распарсить ответ агента",
                    "search_queries": ["React разработчик", "Frontend React", "React developer"],
                    "filters": {
                        "location": None,
                        "min_experience_years": None,
                        "required_skills": ["React", "React.js"]
                    },
                    "analysis_instructions": "Найди кандидатов, которые знают React или React.js.",
                    "requires_refinement": False
                }
                
        except Exception as e:
            print(f"❌ Критическая ошибка при парсинге JSON: {e}")
            print(f"📝 Исходный ответ: {response[:500]}...")
            
            # Создаем дефолтный ответ для поиска React
            return {
                "thought_process": "Поиск React-разработчиков (дефолтный ответ)",
                "search_queries": ["React разработчик", "Frontend React", "React developer"],
                "filters": {
                    "location": None,
                    "min_experience_years": None,
                    "required_skills": ["React"]
                },
                "analysis_instructions": "Проанализируй найденные резюме на знание React.",
                "requires_refinement": False
            }
    
    def _build_filters(self, parsed_response: dict) -> dict:
        """Построение фильтров для ChromaDB на основе ответа агента."""
        filters = {}
        conditions = []
        
        # Фильтр по городу (используем $eq)
        location = parsed_response.get("filters", {}).get("location")
        if location and location.lower() != "null" and location.lower() != "none":
            city = location.lower()
            conditions.append({"location": {"$eq": city}})
            print(f"📍 Фильтр по городу: {city}")
        
        # Фильтр по минимальному опыту (используем $gte)
        min_exp = parsed_response.get("filters", {}).get("min_experience_years")
        if min_exp and min_exp != "null" and min_exp != "none":
            try:
                min_months = int(min_exp) * 12
                conditions.append({"total_experience_months": {"$gte": min_months}})
                print(f"📅 Фильтр по опыту: от {min_exp} лет ({min_months} месяцев)")
            except (ValueError, TypeError):
                pass
        
        # ВАЖНО: ChromaDB не поддерживает $contains, поэтому убираем фильтрацию по навыкам
        # Вместо этого будем искать по эмбеддингам и фильтровать результаты позже
        required_skills = parsed_response.get("filters", {}).get("required_skills", [])
        if required_skills and isinstance(required_skills, list):
            valid_skills = []
            for skill in required_skills[:3]:
                if skill and str(skill).lower() not in ["null", "none"]:
                    valid_skills.append(str(skill).lower())
            
            if valid_skills:
                print(f"🔧 Навыки для поиска (без фильтрации в where): {valid_skills}")
                # Сохраняем навыки в объекте для последующей фильтрации
                self._temp_required_skills = valid_skills
        
        # Формируем условия
        if conditions:
            if len(conditions) > 1:
                filters = {"$and": conditions}
            else:
                filters = conditions[0]
        
        print(f"🔧 Построенные фильтры для ChromaDB: {filters}")
        return filters
    
    async def _search_with_refinement(self, 
                                initial_queries: List[str], 
                                filters: Dict[str, Any],
                                max_results: int = 10) -> List[Dict[str, Any]]:
        """Итеративный поиск с уточнением запросов."""
        all_resumes = []
        seen_ids = set()
        
        # Получаем список требуемых навыков, если есть
        required_skills = getattr(self, '_temp_required_skills', [])
        
        # Первый раунд поиска
        for query in initial_queries:
            if len(all_resumes) >= max_results:
                break
                
            query_emb = self.model.encode(query).tolist()
            results = self.collection.query(
                query_embeddings=[query_emb],
                n_results=min(20, max_results * 3),  # Берем больше, чтобы отфильтровать
                where=filters if filters else None,
                include=["documents", "metadatas"]
            )
            
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                resume_id = meta.get("id", "")
                if resume_id and resume_id not in seen_ids:
                    # Проверяем наличие требуемых навыков в поле all_skills
                    all_skills = meta.get("all_skills", "").lower()
                    
                    # Если есть требования по навыкам, проверяем их
                    if required_skills:
                        has_required_skill = False
                        for skill in required_skills:
                            if skill in all_skills:
                                has_required_skill = True
                                break
                        
                        # Если не нашли нужный навык, пропускаем это резюме
                        if not has_required_skill:
                            continue
                    
                    seen_ids.add(resume_id)
                    all_resumes.append({
                        "id": resume_id,
                        "url": meta.get("url", "").strip(),
                        "position": meta.get("desired_position", ""),
                        "location": meta.get("location", ""),
                        "experience_months": meta.get("total_experience_months", 0),
                        "skills": all_skills,
                        "text": doc
                    })
        
        print(f"🔍 Первый раунд дал {len(all_resumes)} резюме")
        
        # Если нашли достаточно, возвращаем
        if len(all_resumes) >= max_results // 2:
            return all_resumes[:max_results]
        
        # Второй раунд: поиск без фильтров по навыкам
        print("🔍 Пробую fallback (без фильтрации по навыкам)...")
        
        # Ослабляем фильтры: убираем требования по навыкам
        for query in initial_queries:
            if len(all_resumes) >= max_results:
                break
                
            query_emb = self.model.encode(query).tolist()
            results = self.collection.query(
                query_embeddings=[query_emb],
                n_results=min(15, max_results * 2),
                where=filters if filters else None,  # Оставляем только базовые фильтры (город, опыт)
                include=["documents", "metadatas"]
            )
            
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                resume_id = meta.get("id", "")
                if resume_id and resume_id not in seen_ids:
                    seen_ids.add(resume_id)
                    all_resumes.append({
                        "id": resume_id,
                        "url": meta.get("url", "").strip(),
                        "position": meta.get("desired_position", ""),
                        "location": meta.get("location", ""),
                        "experience_months": meta.get("total_experience_months", 0),
                        "skills": meta.get("all_skills", "").lower(),
                        "text": doc
                    })
        
        print(f"✅ Итого найдено {len(all_resumes)} резюме")
        return all_resumes[:max_results]
    
    async def process_query(self, user_query: str) -> str:
        """Основной метод обработки запроса пользователя."""
        
        # === Шаг 1: Агент анализирует запрос и планирует поиск ===
        planning_prompt = f'''Ты — HR-аналитик, который ищет кандидатов по базе резюме.

            Запрос пользователя: "{user_query}"

            Твоя задача:
            1. Проанализировать запрос и выделить ключевые элементы:
            - Технологии/навыки (React, Python, ML и т.д.)
            - Город/локация (Москва, удалённо и т.д.)
            - Требования к опыту (от 3 лет, junior/senior и т.д.)
            - Должность (фронтенд, бэкенд, data scientist и т.д.)

            2. Сформулировать 1-3 поисковых запроса для векторного поиска.

            3. Определить фильтры для уточнения поиска.

            Верни ответ в СТРОГОМ JSON формате:
            {{
            "thought_process": "Краткий анализ запроса",
            "search_queries": ["запрос1", "запрос2"],
            "filters": {{
                "location": null,
                "min_experience_years": null,
                "required_skills": ["React"]
            }},
            "analysis_instructions": "Проанализируй резюме",
            "requires_refinement": false
            }}

            Важно: 
            1. Используй только двойные кавычки
            2. Не ставь запятые после последних элементов
            3. Все строковые значения должны быть в одну строку
            4. Не добавляй комментарии в JSON
            '''
        
        agent_response = await self._call_llm_with_retry(
            planning_prompt,
            system_prompt="Ты — эксперт по поиску IT-специалистов. Будь конкретен и точен."
        )
        
        parsed_response = self._parse_agent_response(agent_response)
        print(f"🤖 Агент проанализировал запрос: {parsed_response.get('thought_process', '')}")
        
        # === Шаг 2: Выполняем поиск с возможным уточнением ===
        filters = self._build_filters(parsed_response)
        resumes = await self._search_with_refinement(
            parsed_response.get("search_queries", [user_query]),
            filters,
            max_results=15
        )
        
        if not resumes:
            return "🔍 По вашему запросу не найдено подходящих резюме."
        
        # === Шаг 3: Готовим контекст для анализа ===
        context_parts = []
        for i, r in enumerate(resumes, 1):
            exp_years = r['experience_months'] // 12
            skills_preview = r['skills'][:150] + "..." if len(r['skills']) > 150 else r['skills']
            
            context_parts.append(f"""
Резюме #{i}:
Должность: {r['position']}
Город: {r['location']}
Опыт: {exp_years} лет
Ключевые навыки: {skills_preview}
Краткое описание: {r['text'][:300]}...
            """.strip())
        
        context = "\n\n".join(context_parts)
        
        # === Шаг 4: Агент анализирует найденные резюме ===
        analysis_prompt = f"""
Запрос пользователя: "{user_query}"

{parsed_response.get("analysis_instructions", "Проанализируй найденные резюме на соответствие запросу.")}

Найденные резюме:
{context}

Твоя задача:
1. Оценить релевантность каждого резюме запросу "{user_query}"
2. Выбрать только те резюме, которые действительно подходят
3. Указать номера выбранных резюме (например: 1, 3, 5)
4. Дать краткое обоснование для каждого выбранного резюме
5. Если ни одно не подходит, так и скажи

**ВНИМАНИЕ**: В своём ответе указывай ТОЛЬКО реальные номера резюме из списка выше (1, 2, 3...).

Формат ответа:
**Анализ:**
[номера резюме через запятую] - краткое обоснование выбора.

**Подходящие кандидаты:**
[номер]: Должность, Город, Опыт, Ключевые навыки
"""
        
        analysis = await self._call_llm_with_retry(
            analysis_prompt,
            system_prompt="Ты — строгий HR-аналитик. Выбирай только действительно подходящих кандидатов."
        )
        
        # === Шаг 5: Извлекаем номера подходящих резюме ===
        relevant_indices = []
        lines = analysis.split('\n')
        
        for line in lines:
            # Ищем строки с перечислением номеров
            if re.search(r'^\d+(?:\s*,\s*\d+)*', line.strip()):
                numbers = re.findall(r'\b(\d+)\b', line)
                relevant_indices.extend([int(num)-1 for num in numbers])
            # Ищем отдельные упоминания номеров
            else:
                numbers = re.findall(r'\bРезюме\s+#?(\d+)\b', line, re.IGNORECASE)
                relevant_indices.extend([int(num)-1 for num in numbers])
        
        # Убираем дубликаты и неверные индексы
        relevant_indices = sorted(set([idx for idx in relevant_indices if 0 <= idx < len(resumes)]))
        
        # === Шаг 6: Формируем финальный ответ ===
        final_answer = f"**Запрос:** {user_query}\n\n"
        final_answer += f"**Анализ агента:**\n{analysis}\n\n"
        
        if relevant_indices:
            final_answer += "🔗 **Ссылки на подходящие резюме:**\n"
            seen_urls = set()
            link_counter = 1
            
            for idx in relevant_indices:
                r = resumes[idx]
                if r['url'] and r['url'] not in seen_urls:
                    seen_urls.add(r['url'])
                    exp_years = r['experience_months'] // 12
                    final_answer += f"{link_counter}. {r['position'] or 'Должность не указана'} "
                    final_answer += f"(г. {r['location'] or 'Город не указан'}, опыт {exp_years} лет)\n"
                    final_answer += f"{r['url']}\n\n"
                    link_counter += 1
        else:
            final_answer += "ℹ️ **Рекомендация:** Агент не нашёл подходящих кандидатов. Попробуйте изменить критерии поиска."
        
        return final_answer