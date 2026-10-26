# 第 3 课参考实现：读 PDF + 切分
# 依赖：pip install pypdf
# 运行：python src/ingest.py   （在项目根目录下）

from pypdf import PdfReader
import os


def load_pdf(path):
    """把 PDF 每一页的文字抽出来，拼成一个整体字符串"""
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def chunk_text(text, chunk_size=500, overlap=50):
    """按固定字符长度切块，相邻块重叠 overlap 个字符"""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


if __name__ == "__main__":
    # 改成你自己的测试文档路径
    pdf_path = os.path.join("data", "document.pdf")
    if not os.path.exists(pdf_path):
        print("请先把测试 PDF 放到 data/document.pdf")
    else:
        text = load_pdf(pdf_path)
        chunks = chunk_text(text)
        print("总字符数：", len(text))
        print("切分块数：", len(chunks))
        if chunks:
            print("第一块前 100 字：", chunks[0][:100])
