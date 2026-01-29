"""
Analyzer for grok-CAIP-210.txt questions
Extracts and converts questions to JavaScript format
"""

import re
import json

def parse_grok_questions(txt_path):
    """Parse questions from grok-CAIP-210.txt"""
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by lessons
    lessons = re.split(r'Lesson \d+:', content)
    
    all_questions = []
    question_count = 0
    
    for lesson_idx, lesson_text in enumerate(lessons[1:], 1):  # Skip intro
        print(f"\n{'='*60}")
        print(f"LESSON {lesson_idx}")
        print(f"{'='*60}")
        
        # Extract lesson title
        title_match = re.search(r'^([^\n]+)', lesson_text)
        lesson_title = title_match.group(1).strip() if title_match else f"Lesson {lesson_idx}"
        print(f"Title: {lesson_title}")
        
        # Find all questions in this lesson
        # Pattern: Questão X.Y (Type)
        question_pattern = r'Questão (\d+\.\d+) \((.*?)\)(.*?)(?=Questão \d+\.\d+|Lesson \d+:|$)'
        questions = re.findall(question_pattern, lesson_text, re.DOTALL)
        
        print(f"Found {len(questions)} questions")
        
        for q_num, q_type, q_content in questions:
            question_count += 1
            
            # Extract question text
            lines = q_content.strip().split('\n')
            
            # Find scenario if exists
            scenario = ""
            if 'Cenário:' in q_content:
                scenario_match = re.search(r'Cenário:(.*?)(?=Qual|Por que|Discuta|Explique|Verdadeiro)', q_content, re.DOTALL)
                if scenario_match:
                    scenario = scenario_match.group(1).strip()
            
            # Find main question
            question_text = ""
            for line in lines:
                if line.strip() and not line.startswith('Cenário:') and not line.startswith('A)') and not line.startswith('B)') and not line.startswith('C)') and not line.startswith('D)') and not line.startswith('Resposta') and not line.startswith('Justificativa:'):
                    if 'Qual' in line or 'Por que' in line or 'O que' in line or 'Como' in line or 'Quando' in line or 'Verdadeiro' in line or 'Discuta' in line or 'Explique' in line:
                        question_text = line.strip()
                        break
            
            # Extract options (for multiple choice)
            options = []
            option_pattern = r'([A-D])\) (.+?)(?=\n[A-D]\)|Resposta correta:|$)'
            option_matches = re.findall(option_pattern, q_content, re.DOTALL)
            for opt_letter, opt_text in option_matches:
                options.append(opt_text.strip())
            
            # Extract correct answer
            correct_answer = None
            correct_match = re.search(r'Resposta correta: ([A-D]|Verdadeiro|Falso|.*?)\n', q_content)
            if correct_match:
                correct_text = correct_match.group(1).strip()
                if correct_text in ['A', 'B', 'C', 'D']:
                    correct_answer = ord(correct_text) - ord('A')  # Convert to 0-indexed
                elif correct_text == 'Verdadeiro':
                    correct_answer = 0  # True
                elif correct_text == 'Falso':
                    correct_answer = 1  # False
            
            # Extract justification/explanation
            explanation = ""
            justif_match = re.search(r'Justificativa:(.*?)(?=Questão|Lesson|$)', q_content, re.DOTALL)
            if justif_match:
                explanation = justif_match.group(1).strip()
            
            # Map lesson to domain
            domain_map = {
                1: 1,  # AI & ML Fundamentals
                2: 2,  # Data Preparation
                3: 3,  # Training & Tuning
                4: 3,  # Linear Regression (Training)
                5: 3,  # Forecasting (Training)
                6: 3,  # Classification (Training)
                7: 3,  # Clustering (Training)
                8: 3,  # Decision Trees (Training)
                9: 3,  # SVM (Training)
                10: 3, # Neural Networks (Training)
                11: 4, # Operationalizing (MLOps)
                12: 4  # Maintaining (MLOps)
            }
            
            domain = domain_map.get(lesson_idx, 1)
            
            # Only add if it's multiple choice with 4 options
            if len(options) == 4 and correct_answer is not None:
                question_obj = {
                    'id': 100 + question_count,  # Start from 101 to avoid conflicts
                    'domain': domain,
                    'lesson': lesson_idx,
                    'type': q_type,
                    'scenario': scenario,
                    'question': question_text,
                    'options': options,
                    'correct': correct_answer,
                    'explanation': explanation
                }
                all_questions.append(question_obj)
                print(f"  ✅ Q{q_num}: {q_type} - {len(options)} options")
            else:
                print(f"  ⚠️  Q{q_num}: {q_type} - Skipped (not standard format)")
    
    return all_questions

def generate_js_file(questions, output_path):
    """Generate JavaScript file with questions"""
    
    domain_names = {
        1: "AI & ML Fundamentals",
        2: "Data Preparation",
        3: "Training & Tuning",
        4: "MLOps & Production"
    }
    
    js_content = "// CAIP-210 Advanced Questions from grok-CAIP-210.txt\n"
    js_content += "// High-quality contextual and analytical questions\n\n"
    js_content += "const questionsExtra3 = [\n"
    
    for i, q in enumerate(questions):
        js_content += "    {\n"
        js_content += f"        id: {q['id']},\n"
        js_content += f"        domain: {q['domain']},\n"
        js_content += f"        domainName: \"{domain_names[q['domain']]}\",\n"
        
        # Combine scenario and question if scenario exists
        full_question = q['question']
        if q['scenario']:
            full_question = f"Cenário: {q['scenario']} {q['question']}"
        
        js_content += f"        question: \"{full_question.replace(chr(34), chr(39))}\",\n"
        js_content += "        options: [\n"
        for opt in q['options']:
            js_content += f"            \"{opt.replace(chr(34), chr(39))}\",\n"
        js_content += "        ],\n"
        js_content += f"        correct: {q['correct']},\n"
        js_content += f"        explanation: \"{q['explanation'].replace(chr(34), chr(39))}\"\n"
        js_content += "    }" + ("," if i < len(questions) - 1 else "") + "\n"
    
    js_content += "];\n"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(js_content)
    
    print(f"\n✅ Generated {output_path} with {len(questions)} questions")

if __name__ == "__main__":
    print("🔍 Analyzing grok-CAIP-210.txt...")
    
    questions = parse_grok_questions("grok-CAIP-210.txt")
    
    print(f"\n{'='*60}")
    print(f"📊 SUMMARY")
    print(f"{'='*60}")
    print(f"Total questions extracted: {len(questions)}")
    
    # Count by domain
    domain_counts = {}
    for q in questions:
        domain_counts[q['domain']] = domain_counts.get(q['domain'], 0) + 1
    
    print("\nBy domain:")
    for domain, count in sorted(domain_counts.items()):
        print(f"  Domain {domain}: {count} questions")
    
    # Generate JS file
    if questions:
        generate_js_file(questions, "exam-prep/questions-extra3.js")
        print("\n✅ Conversion completed!")
    else:
        print("\n⚠️  No questions extracted")
