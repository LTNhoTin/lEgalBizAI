# LegalBizAI_project (LLM)

## Mục tiêu

Dự án LegalBizAI nhằm mục tiêu phát triển một hệ thống trí tuệ nhân tạo (AI) chuyên về luật doanh nghiệp, có khả năng tự động truy vấn và trả lời các câu hỏi về văn bản pháp luật.

## Các bước triển khai

### 1. Cào dữ liệu từ thuvienphapluat.vn

Chúng tôi bắt đầu bằng việc thu thập dữ liệu từ trang web thuvienphapluat.vn, bao gồm:

- Các văn bản gốc
- Các văn bản hợp nhất
- Các nghị định

### 2. Phân chia dữ liệu

Dữ liệu sau khi được cào về sẽ được chia nhỏ theo cấu trúc:

- Chương
- Mục
- Điều, khoản

Mỗi phần dữ liệu sẽ được chia thành các đoạn nhỏ hơn (chunk), với cấu trúc như sau:

```
{
  "id": "mã định danh của đoạn văn bản",
  "title": "Điều số mấy",
  "passage": "nội dung của điều"
}
```

### 3. Thu thập bộ dữ liệu câu hỏi và trả lời 

Chúng tôi đã tiến hành thu thập một bộ dữ liệu câu hỏi và trả lời để chuẩn bị cho việc xây dựng bộ testset. Bộ dữ liệu này sẽ giúp chúng tôi đánh giá hiệu quả của hệ thống AI trong việc truy vấn và trả lời câu hỏi.
```
{
  "question": "Câu hỏi ",
  "answer": "Câu trả lời ",
  "type_question": "Phân loại câu hỏi ",
  "chunk_ids": "Gán chunk theo id tương ứng "

}
```

### 4. Phát triển công cụ hỗ trợ

Để dễ dàng gán mã định danh (chunk_ids) cho các file testset, chúng tôi đã phát triển một số công cụ nhỏ. Những công cụ này giúp chúng tôi bán tự động hoá quá trình gán mã định danh, đảm bảo tính nhất quán và chính xác trong bộ testset.

```
## General Tools
generaltools
  ├── tools_checkID/new
  │   └── clausewarticle.py
  └── tools_testset
      ├── autotypeqs.py
      ├── deleterange.py
      ├── duplicate.py
      ├── keepqs.py
      └── typeqs.ipynb

```

### 5. Cào và Xử lý Dữ liệu dành cho finetune

* **Nguồn:** 36,000 liên kết câu hỏi từ Thư viện Pháp luật.
* **Kết quả:** Hơn 120,000 câu hỏi và câu trả lời (sau xử lý còn 105,000).

### 6. Fine-tuning Mô hình

* **Mô hình:** Vistral 7B
* **Mục tiêu:** Cải thiện khả năng truy vấn và trả lời của hệ thống.

## Công việc Tương lai

* Hoàn thiện mô hình Retrieval-Augmented Generation (RAG).
* Tiếp tục fine-tune mô hình Vistral 7B.
* Phát triển giao diện người dùng thân thiện.
* Triển khai hệ thống trên môi trường thực tế.
