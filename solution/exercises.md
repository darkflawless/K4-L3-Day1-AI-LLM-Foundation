# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi temperature = 0.0, mô hình đưa ra câu trả lời mang tính tất định (deterministic), tập trung vào các dữ kiện phổ biến nhất (như xuất khẩu cà phê/hạt điều) và lặp lại giống nhau giữa các lần chạy. Khi tăng lên 0.5 - 1.0, câu trả lời bắt đầu đa dạng hóa văn phong và đề cập đến những góc nhìn mới lạ hơn (như hang Sơn Đoòng, văn hóa ẩm thực). Ở mức 1.5, văn phong trở nên cực kỳ sáng tạo và phóng khoáng, nhưng ngữ pháp có phần bất quy tắc hơn và dễ xuất hiện hiện tượng bịa đặt (hallucination).

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ đặt temperature thấp, trong khoảng từ 0.0 đến 0.2 (tối đa 0.3). Chatbot hỗ trợ khách hàng đòi hỏi tính chính xác, nhất quán và độ tin cậy tuyệt đối về chính sách, giá cả và quy trình đổi trả; việc giữ temperature thấp giúp hạn chế tối đa nguy cơ mô hình "sáng tạo" thông tin sai lệch gây rủi ro pháp lý hoặc thiệt hại cho doanh nghiệp.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Tổng lượng token output mỗi ngày là 10.000 x 3 x 350 = 10.500.000 token (10.500k token). Chi phí output của GPT-4o là 10.500 x $0.010 = $105/ngày, trong khi GPT-4o-mini là 10.500 x $0.0006 = $6.3/ngày; do đó GPT-4o đắt hơn GPT-4o-mini khoảng 16.7 lần (chênh lệch gần $3.000/tháng).
> - Trường hợp GPT-4o xứng đáng: Phân tích hợp đồng pháp lý, tư vấn tài chính chuyên sâu, hoặc giải các bài toán lập trình/logic phức tạp nhiều bước đòi hỏi suy luận cao cấp.
> - Trường hợp nên dùng mini: Chatbot trả lời FAQ dựa trên dữ liệu có sẵn, tóm tắt tin tức ngắn, phân loại ý định người dùng (intent classification), hoặc trích xuất thông tin dạng bảng đơn giản.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Phản hồi của 'giáo viên tiểu học' ngắn gọn, dùng từ ngữ mộc mạc và sử dụng hình ảnh ẩn dụ quen thuộc như 'cuốn sổ tay chung của cả lớp mà ai cũng có một bản chép giống nhau để không ai nói dối được'. Ngược lại, 'chuyên gia tài chính' sử dụng câu từ trang trọng, dài hơn và chứa nhiều thuật ngữ chuyên sâu (sổ cái phân tán DLT, mật mã học, cơ chế đồng thuận PoW/PoS, tính bất biến). Điều này chứng minh system prompt đóng vai trò như một bộ lọc điều hướng ngữ nghĩa (semantic steering), định hình toàn bộ phong cách, góc nhìn, và không gian từ vựng mà mô hình sẽ sử dụng.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Với đoạn văn tiếng Việt 100 từ, công thức ước lượng thô cho ra khoảng 133 token (100 / 0.75), trong khi tiktoken đếm thực tế thường dao động từ 180 đến 220 token (chênh lệch thực tế cao hơn từ 35% đến 65%). Tiếng Việt tốn nhiều token hơn tiếng Anh vì các thuật toán tokenization (như BPE) được huấn luyện chủ yếu trên tập dữ liệu tiếng Anh; các từ tiếng Việt có dấu thanh và ghép âm thường không nằm trọn vẹn trong từ điển token mà bị bẻ nhỏ thành nhiều mảnh subword hoặc từng byte riêng lẻ.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất trong các ứng dụng có tương tác trực tiếp với con người (như chatbot, trợ lý ảo terminal, ứng dụng hỏi đáp) vì nó tối ưu thời gian phản hồi đầu tiên (Time to First Token - TTFT), giúp người dùng thấy văn bản xuất hiện ngay sau 1-2 giây thay vì phải nhìn màn hình chờ đợi 10-20 giây. Ngược lại, non-streaming phù hợp hơn khi xử lý tác vụ ngầm theo lô (batch processing), các endpoint API trả về JSON có cấu trúc (Function Calling / Tool Calling) cần dữ liệu nguyên vẹn để code parse, hoặc khi cần kiểm duyệt toàn bộ nội dung (moderation) trước khi chuyển tiếp cho người dùng.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff giúp giãn cách thời gian giữa các lần thử lại ngày càng xa (ví dụ: 0.1s, 0.2s, 0.4s...), giúp giảm tải áp lực cho máy chủ và tạo điều kiện cho hệ thống có thời gian giải phóng tài nguyên để hồi phục. Nếu hàng nghìn client cùng dùng delay cố định (như 1 giây), toàn bộ các client sẽ đồng loạt gửi request trở lại vào đúng cùng một thời điểm, gây ra hiện tượng 'cơn bão retry' (thundering herd problem) làm máy chủ vừa chớm khởi động lại ngay lập tức bị sập tiếp.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Persona tôi chọn là: "Bạn là trợ giảng AI thân thiện của khóa học, luôn giải thích dễ hiểu, trả lời ngắn gọn dưới 3 đoạn văn bằng tiếng Việt chuẩn mực và đưa ra ví dụ trực quan."
> - Cụm từ "trả lời ngắn gọn dưới 3 đoạn văn" rất quan trọng để kiểm soát độ dài phản hồi, vừa tiết kiệm token vừa đảm bảo giao diện dòng lệnh (CLI) không bị tràn màn hình, giúp người học dễ theo dõi.
> - Cụm từ "bằng tiếng Việt chuẩn mực" định hướng mô hình duy trì ngôn ngữ nhất quán, tránh việc pha trộn tiếng Anh không cần thiết khi người dùng hỏi các thuật ngữ kỹ thuật.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất là cơ chế cửa sổ trượt chỉ giữ 3 lượt hội thoại gần nhất (6 message), khiến trợ lý quên sạch các thông tin hoặc ràng buộc mà người dùng đã đề cập trước đó trong phiên chat dài.
> - Cải thiện đề xuất: Tích hợp cơ chế "Tóm tắt ngữ cảnh" (Context Summarization).
> - Cách triển khai: Khi lịch sử vượt quá 6 message, ta dùng một model nhỏ (như GPT-4o-mini) tóm tắt các lượt hội thoại cũ thành một đoạn văn ngắn gọn (chỉ tốn khoảng 30-50 token) rồi gắn đoạn tóm tắt này vào đầu system prompt; nhờ đó trợ lý vừa nhớ được toàn bộ diễn biến quá khứ vừa không làm bùng nổ số lượng token của request.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
