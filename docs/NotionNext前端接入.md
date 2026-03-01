# NotionNext 前端接入示例

## 1. 环境变量
在 NotionNext 项目中新增：

```env
NEXT_PUBLIC_KB_AI_BASE_URL=http://localhost:8000
```

## 2. 页面示例（`pages/ai.tsx`）

```tsx
import { useState } from 'react'

const baseUrl = process.env.NEXT_PUBLIC_KB_AI_BASE_URL

export default function AIPage() {
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState('')
  const [sources, setSources] = useState<string[]>([])

  const ask = async () => {
    const res = await fetch(`${baseUrl}/api/v1/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, session_id: 'web-user' })
    })
    const data = await res.json()
    setAnswer(data.answer)
    setSources(data.sources || [])
  }

  return (
    <main style={{ maxWidth: 900, margin: '40px auto' }}>
      <h1>知识库问答</h1>
      <textarea value={question} onChange={(e) => setQuestion(e.target.value)} rows={4} />
      <button onClick={ask}>提问</button>
      <pre>{answer}</pre>
      <ul>
        {sources.map((s) => (
          <li key={s}>{s}</li>
        ))}
      </ul>
    </main>
  )
}
```

## 3. 流式 SSE 示例

```ts
const res = await fetch(`${baseUrl}/api/v1/chat/stream`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ question, session_id: 'web-user' })
})

const reader = res.body?.getReader()
const decoder = new TextDecoder()
let buffer = ''

while (reader) {
  const { done, value } = await reader.read()
  if (done) break
  buffer += decoder.decode(value, { stream: true })
  // 按 SSE 协议解析 data: 行
}
```

