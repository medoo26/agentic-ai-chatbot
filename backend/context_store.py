from collections import defaultdict

# يخزن السياق مؤقت في الذاكرة
SESSION_CONTEXT = defaultdict(list)

MAX_CONTEXT_MESSAGES = 6  # آخر 6 رسائل فقط

def add_to_context(session_id, role, content):
    SESSION_CONTEXT[session_id].append({
        "role": role,
        "content": content
    })

    # نحتفظ بآخر 6 رسائل فقط
    SESSION_CONTEXT[session_id] = SESSION_CONTEXT[session_id][-MAX_CONTEXT_MESSAGES:]


def get_context(session_id):
    return SESSION_CONTEXT.get(session_id, [])
