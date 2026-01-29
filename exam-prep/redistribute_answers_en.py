import json
import random
import re

def redistribute_answers(input_file, output_file):
    """Redistribute answers in a questions file to achieve balanced distribution."""
    print(f"\nProcessing: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all question blocks
    pattern = r'(\{\s*id:\s*\d+,.*?explanation:.*?\})'
    questions = re.findall(pattern, content, re.DOTALL)
    
    print(f"Found {len(questions)} questions")
    
    # Count current distribution
    current_dist = {0: 0, 1: 0, 2: 0, 3: 0}
    for q_str in questions:
        correct_match = re.search(r'correct:\s*(\d)', q_str)
        if correct_match:
            current_dist[int(correct_match.group(1))] += 1
    
    print(f"Current distribution: A={current_dist[0]}, B={current_dist[1]}, C={current_dist[2]}, D={current_dist[3]}")
    
    # Create target distribution (rotate through 0,1,2,3)
    target_distribution = []
    for i in range(len(questions)):
        target_distribution.append(i % 4)
    
    random.shuffle(target_distribution)  # Randomize the distribution
    
    # Process each question
    new_questions = []
    for i, q_str in enumerate(questions):
        # Extract question components
        id_match = re.search(r'id:\s*(\d+)', q_str)
        options_match = re.search(r'options:\s*\[(.*?)\]', q_str, re.DOTALL)
        correct_match = re.search(r'correct:\s*(\d)', q_str)
        
        if not all([id_match, options_match, correct_match]):
            new_questions.append(q_str)
            continue
        
        qid = int(id_match.group(1))
        current_correct = int(correct_match.group(1))
        target_correct = target_distribution[i]
        
        # Parse options
        options_str = options_match.group(1)
        options = re.findall(r'"([^"]*)"', options_str)
        
        if len(options) != 4:
            new_questions.append(q_str)
            continue
        
        # Shuffle options to place correct answer at target position
        correct_option = options[current_correct]
        other_options = [opt for j, opt in enumerate(options) if j != current_correct]
        random.shuffle(other_options)
        
        new_options = other_options[:target_correct] + [correct_option] + other_options[target_correct:]
        
        # Rebuild options string
        new_options_str = ',\n            '.join([f'"{opt}"' for opt in new_options])
        
        # Replace in question string
        new_q_str = re.sub(r'options:\s*\[.*?\]', f'options: [\n            {new_options_str}\n        ]', q_str, flags=re.DOTALL)
        new_q_str = re.sub(r'correct:\s*\d', f'correct: {target_correct}', new_q_str)
        
        new_questions.append(new_q_str)
    
    # Rebuild the file
    new_content = content
    for old_q, new_q in zip(questions, new_questions):
        new_content = new_content.replace(old_q, new_q, 1)
    
    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    # Count new distribution
    new_dist = {0: 0, 1: 0, 2: 0, 3: 0}
    for q_str in new_questions:
        correct_match = re.search(r'correct:\s*(\d)', q_str)
        if correct_match:
            new_dist[int(correct_match.group(1))] += 1
    
    print(f"New distribution: A={new_dist[0]}, B={new_dist[1]}, C={new_dist[2]}, D={new_dist[3]}")
    print(f"✓ Saved to: {output_file}\n")

if __name__ == "__main__":
    files_to_process = [
        ("questions-en.js", "questions-en.js"),
        ("questions-extra-en.js", "questions-extra-en.js"),
        ("questions-extra2-en.js", "questions-extra2-en.js"),
    ]
    
    for input_file, output_file in files_to_process:
        try:
            redistribute_answers(input_file, output_file)
        except Exception as e:
            print(f"Error processing {input_file}: {e}")
    
    print("\n✅ All English files processed!")
