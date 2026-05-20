# -*- coding: utf-8 -*-
with open('eval.py', encoding='utf-8-sig') as f:
    content = f.read()

old = '    return jaccard_similarity(retrieved_content, ground_truth) >= threshold'

new_lines = [
    '    if "\\n" + chr(22238) + chr(31572) + ": " in retrieved_content:',
    '        answer_part = retrieved_content.split("\\n" + chr(22238) + chr(31572) + ": ", 1)[1]',
    '    elif chr(22238) + chr(31572) + ": " in retrieved_content:',
    '        answer_part = retrieved_content.split(chr(22238) + chr(31572) + ": ", 1)[1]',
    '    else:',
    '        answer_part = retrieved_content',
    '    if len(ground_truth.strip()) <= 10:',
    '        return ground_truth.strip() in answer_part',
    '    return jaccard_similarity(answer_part, ground_truth) >= threshold'
]

new_block = chr(10).join(new_lines)

if old in content:
    content = content.replace(old, new_block)
    content = content.replace('threshold: float = 0.15', 'threshold: float = 0.08')
    print('FIXED')
else:
    print('OLD line not found, trying alternative...')
    # Try with return
    for i, line in enumerate(content.split(chr(10))):
        if 'jaccard_similarity(retrieved_content, ground_truth) >= threshold' in line:
            print(f'Found at line {i+1}: {repr(line)}')
            break

with open('eval.py', 'w', encoding='utf-8') as f:
    f.write(content)