import json
import re
import os
from tqdm import tqdm

def parse_experience_to_months(exp_str: str) -> int:
    if not exp_str:
        return 0
    exp_str = re.sub(r'\s+', '', exp_str.lower())
    years = int(re.search(r'(\d+)лет', exp_str).group(1)) if re.search(r'(\d+)лет', exp_str) else 0
    months = int(re.search(r'(\d+)месяц', exp_str).group(1)) if re.search(r'(\d+)месяц', exp_str) else 0
    return years * 12 + months

def extract_skills_from_experience(resume: dict) -> set:
    """Извлекает навыки из блоков вроде 'Уровни владения навыкамиJavaScriptVue.js...'"""
    skills = set()
    for exp in resume.get("experience_details", []):
        period = exp.get("period", "")
        # Ищем: Уровни владения навыкамиJavaScriptVue.jsNuxt.js...
        match = re.search(r'уровни владения навыками([a-zа-яё0-9\s]+)', period.lower())
        if match:
            raw_skills = match.group(1)
            found_skills = re.findall(r'[a-zа-яё0-9]+(?:\.[a-zа-яё0-9]+)?', raw_skills)
            for s in found_skills:
                if len(s) > 1:
                    skills.add(s.title())
    return skills

def extract_tech_keywords(text: str) -> list:
    """Извлекает технологические ключевые слова из текста описания."""
    if not text:
        return []
    tech_patterns = {
        'react': ['react', 'react.js', 'reactjs'],
        'vue': ['vue', 'vue.js', 'vuejs', 'nuxt', 'nuxt.js'],
        'angular': ['angular'],
        'jquery': ['jquery'],
        'javascript': ['javascript', 'js', 'ecmascript'],
        'typescript': ['typescript', 'ts'],
        'python': ['python'],
        'django': ['django'],
        'flask': ['flask'],
        'fastapi': ['fastapi'],
        'node': ['node', 'node.js', 'nodejs'],
        'java': ['java'],
        'spring': ['spring'],
        'c#': ['c#', 'c sharp'],
        'php': ['php'],
        'laravel': ['laravel'],
        'ruby': ['ruby'],
        'rails': ['rails'],
        'go': ['go', 'golang'],
        'docker': ['docker'],
        'kubernetes': ['kubernetes', 'k8s'],
        'aws': ['aws', 'amazon web services'],
        'sql': ['sql', 'postgresql', 'mysql', 'oracle'],
        'mongodb': ['mongodb', 'mongo'],
        'redis': ['redis'],
        'git': ['git', 'github', 'gitlab'],
        'html': ['html'],
        'css': ['css', 'sass', 'scss', 'less'],
        'webpack': ['webpack'],
        'redux': ['redux'],
        'graphql': ['graphql'],
        'rest': ['rest', 'rest api'],
        'websocket': ['websocket'],
        'ml': ['машинное обучение', 'ml', 'machine learning'],
        'ai': ['искусственный интеллект', 'ai', 'artificial intelligence'],
        'data science': ['data science']
    }
    text_lower = text.lower()
    found_tech = set()
    for tech, patterns in tech_patterns.items():
        for pattern in patterns:
            if pattern in text_lower:
                found_tech.add(tech.title())
                break  # Не ищем другие синонимы той же технологии
    return list(found_tech)

def extract_descriptions(resume: dict) -> str:
    seen = set()
    descs = []
    for exp in resume.get("experience_details", []):
        d = exp.get("description", "").strip()
        if d and d not in seen:
            d = re.sub(r'\s+', ' ', d).replace('\r', '').replace('\n', ' ')
            descs.append(d)
            seen.add(d)
    return " ".join(descs)

def extract_all_skills(resume: dict) -> list:
    skills = set()
    # 1. Основные навыки из раздела skills
    for s in (resume.get("skills") or []):
        if s and len(s.strip()) > 1:
            skills.add(s.strip().title())
    # 2. Навыки из структурированного блока skills_by_level
    for level_skills in (resume.get("skills_by_level") or {}).values():
        for s in (level_skills or []):
            if s and len(s.strip()) > 1:
                skills.add(s.strip().title())
    # 3. Навыки из описания опыта работы
    skills.update(extract_skills_from_experience(resume))
    # 4. Ключевые слова из текста "О себе"
    about_text = resume.get("additional_info", {}).get("about", "")
    if about_text:
        skills.update(extract_tech_keywords(about_text))
    # 5. Ключевые слова из текста описания опыта (самое важное!)
    exp_text = extract_descriptions(resume)
    if exp_text:
        skills.update(extract_tech_keywords(exp_text))
    return sorted(skills)

def extract_education(resume: dict) -> str:
    edu_parts = []
    for item in resume.get("education_details", {}).get("higher", []):
        inst = item.get("institution", "").strip()
        details = item.get("details", "").strip()
        if inst or details:
            edu_parts.append(f"{inst}: {details}" if details else inst)
    return "; ".join(edu_parts) if edu_parts else ""

def clean_location(loc: str) -> str:
    if not loc:
        return ""
    return re.split(r'[,\–—]', loc)[0].strip()

def process_resumes(input_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    resumes = data.get("resumes", [])
    print(f"Найдено {len(resumes)} резюме. Обработка...")
    documents = []
    metadata_list = []
    for resume in tqdm(resumes):
        res_id = resume.get("id", "")
        url = resume.get("url", "").strip()
        pos = resume.get("desired_position", "")
        loc = clean_location(resume.get("location_relocation") or resume.get("personal_info", {}).get("location", ""))
        exp_months = parse_experience_to_months(resume.get("total_experience", ""))
        # === ИЗВЛЕЧЕНИЕ НАВЫКОВ ===
        skills_list = extract_all_skills(resume)
        # === ОБРАБОТКА ОПИСАНИЯ ОПЫТА ===
        experience_desc = extract_descriptions(resume)
        # === ОБРАБОТКА ОБРАЗОВАНИЯ ===
        edu = extract_education(resume)
        # === ФОРМИРОВАНИЕ БОГАТОГО ТЕКСТОВОГО ДОКУМЕНТА ===
        doc_parts = []
        # 1. Ключевая информация первой строкой (для релевантности)
        if pos:
            doc_parts.append(f"Ищу позицию: {pos}")
        if skills_list:
            # Навыки - В НАЧАЛЕ документа для повышения релевантности
            doc_parts.append(f"Ключевые навыки: {', '.join(skills_list[:20])}")
        # 2. Локация и опыт
        if loc:
            doc_parts.append(f"Локация: {loc}")
        if exp_months > 0:
            years = exp_months // 12
            months = exp_months % 12
            exp_text = f"{years} год{'а' if years % 10 in [2,3,4] and years % 100 not in [12,13,14] else 'ов'}" if years > 0 else ""
            if months > 0:
                exp_text += f" {months} месяц{'а' if months % 10 in [2,3,4] and months % 100 not in [12,13,14] else 'ев'}" if months > 0 else ""
            doc_parts.append(f"Опыт работы: {exp_text.strip()}")
        # 3. Описание опыта (основной контент)
        if experience_desc:
            clean_desc = re.sub(r'\s+', ' ', experience_desc)
            if len(clean_desc) > 800:
                clean_desc = clean_desc[:800] + "..."
            doc_parts.append(f"Опыт работы: {clean_desc}")
        # 4. Образование и дополнительная информация
        if edu:
            doc_parts.append(f"Образование: {edu}")
        about = resume.get("additional_info", {}).get("about", "")
        if about and len(about) > 30:
            clean_about = re.sub(r'\s+', ' ', about)
            if len(clean_about) > 200:
                clean_about = clean_about[:200] + "..."
            doc_parts.append(f"О себе: {clean_about}")
        # 5. Специальность
        specialty = resume.get("specialty_category", "")
        if specialty:
            doc_parts.append(f"Специализация: {specialty}")
        # Формируем итоговый текст документа
        doc_text = "\n".join(doc_parts)
        # Минимальная проверка качества документа
        if len(doc_text.strip()) < 100:
            doc_text = f"Кандидат: {pos or 'не указана'}. Навыки: {', '.join(skills_list[:5]) if skills_list else 'не указаны'}. Город: {loc or 'не указан'}."
        # Сохраняем документ
        documents.append({"id": res_id, "text": doc_text})
        # Сохраняем метаданные (с полным списком навыков для фильтрации)
        metadata_list.append({
            "id": res_id,
            "url": url,
            "desired_position": pos,
            "location": loc,
            "total_experience_months": exp_months,
            "skills": skills_list,  # Полный список навыков
            "top_5_skills": skills_list[:5] if skills_list else [],
            "specialty_category": specialty,
            "education": edu
        })
    # Сохранение результатов
    with open(os.path.join(output_dir, "documents.jsonl"), "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    with open(os.path.join(output_dir, "metadata.jsonl"), "w", encoding="utf-8") as f:
        for meta in metadata_list:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    with open(os.path.join(output_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump({
            "total": len(resumes),
            "with_skills": sum(1 for m in metadata_list if m["skills"]),
            "avg_skills_per_resume": sum(len(m["skills"]) for m in metadata_list) / len(metadata_list) if metadata_list else 0,
            "sample_skills": skills_list[:10] if skills_list else []  # Пример извлеченных навыков
        }, f, ensure_ascii=False, indent=2)
    print(f"✅ Готово! Обработано {len(resumes)} резюме.")
    print(f"📊 Статистика: среднее количество навыков на резюме: {sum(len(m['skills']) for m in metadata_list) / len(metadata_list):.1f}")

if __name__ == "__main__":
    process_resumes("./data/resumes.json", "./data/processed")