# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 14h00–18h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.7, 1.2 và 1.8 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Hà Nội."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi? Ở mức nào phản hồi bắt đầu
kém mạch lạc ** 
> *Khi temperature tăng từ 0.0 lên 1.8, câu trả lời thay đổi từ thông tin mang tính chuẩn mựcphổ biến (chẳng hạn như 36 phố phường) sang phong phú, đa dạng hơn (Con đường Gốm Sứ, Hồ Gươm, Cầu Long Biên). Mặc dù ở mức 1.8 nội dung vẫn đọc hiểu được, nhưng phản hồi bắt đầu kém mạch lạc và sai lệch thông tin lịch sử (như khẳng định Gustave Eiffel thiết kế cầu Long Biên  cho thấy AI bắt đầu ngẫu hứng và tưởng tượng hơn)*

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho trợ lý soạn thảo hợp đồng pháp lý,
và bao nhiêu cho trợ lý viết slogan quảng cáo? Giải thích khác biệt.**
> *Soạn thảo hợp đồng pháp lý chọn temperature = 0.0. Viết slogan quảng cáo chọn temperature  0.8 đến 1.0. Lý do vì hợp đồng pháp lý yêu cầu tính chính xác tuyệt đối, chuẩn xác về mặt thuật ngữ và độ nhất quán cao giữa các lần sinh văn bản (tránh hiện tượng sáng tạo hay tự suy diễn luật). Trong khi đó, slogan quảng cáo đòi hỏi sự đa dạng, mới mẻ, độc đáo và giàu cảm xúc, nên mức ngẫu nhiên cao sẽ giúp AI bứt phá khỏi các khuôn mẫu lối mòn để tạo ra nhiều ý tưởng phong phú hơn*

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 20.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 2 lần,
mỗi lần trung bình ~500 token đầu ra.

**Ước tính chi phí mỗi ngày của model lớn so với model nhỏ cho workload này
(dựa trên bảng giá trong template). Nêu một trường hợp model lớn xứng đáng
với chi phí và một trường hợp model nhỏ là lựa chọn đúng:**
> *Ước tính chi phí: Tổng token đầu ra mỗi ngày: 20.000 x 2 x 500 = 20.000.000 tokens (20.000k tokens). Model lớn (GPT-4o): 20.000 x 0.0100 USD = 200 USD/ngày. Model nhỏ (GPT-4o-mini)là 20.000 x 0.0006 USD = 12 USD/ngày (tiết kiệm hơn khoảng 16.6 lần). Model lớn xứng đáng với chi phí khi hệ thống thực hiện các tác vụ phức tạp đòi hỏi khả năng lập luận logic cao, phân tích đa bước hoặc tư vấn chuyên sâu (lập trình hệ thống phức tạp, phân tích báo cáo tài chính). Lựa chọn Model nhỏ khi hệ thống xử lý các tác vụ đơn giản,lặp lại với tần suất cao (ví dụ: phân loại ý định người dùng, kiểm tra lỗi chính tả)*

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích máy học (machine learning) là gì?"** nhưng hai system prompt
khác nhau:
- "Bạn là một nhà thơ, trả lời mọi thứ bằng hình ảnh ví von, tránh thuật ngữ."
- "Bạn là kỹ sư phần mềm senior, trả lời chính xác, có ví dụ code khi phù hợp."

**Hai phản hồi khác nhau như thế nào (giọng văn, độ dài, mức kỹ thuật)?
Từ đó rút ra system prompt điều khiển được những khía cạnh nào của phản hồi?**
(3–4 câu)
> *Phản hồi của nhà thơ mang giọng văn giàu chất thơ, sử dụng hình ảnh so sánh sinh động, độ dài vừa phải và hoàn toàn không dùng thuật ngữ kỹ thuật. Trong khi đó, phản hồi của kỹ sư senior có phong cách chuyên nghiệp, chính xác, độ dài chi tiết hơn và đi thẳng vào bản chất thuật ngữ kèm ví dụ code Python minh họa. System prompt có thể điều khiển trực tiếp các khía cạnh cốt lõi của phản hồi như tông giọng, độ dài,định dạng văn bản,... và có mức độ chuyên sâu kỹ thuật hơn*

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~150 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Nếu dùng ước lượng thô để dự
toán ngân sách API cho ứng dụng tiếng Việt, bạn sẽ dự toán thiếu hay thừa —
và vì sao?**
> *Với đoạn văn mẫu 183 từ, công thức ước lượng thô (số từ / 0.75) tính ra 244.0 token, còn count_tokens(tiktoken) trả về 201 token, chênh lệch thực tế là 17.6%. Với riêng đoạn văn này, công thức thô bị dự toán THỪA vì hệ số 1/0.75 (~1.33 token/từ) vốn được thiết kế cho tiếng Anh, trong khi các từ đơn tiếng Việt ngắn chỉ tốn ~1.1 token/từ. Tuy nhiên xét rộng ra, do tiếng Việt có nhiều dấu thanh và từ ghép làm tokenizer dễ bị tách thành nhiều sub-word tokens, nên việc dùng tiktoken đo trực tiếp để dự toán ngân sách chính xác hơn*

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Xét ba ứng dụng: (a) chatbot văn bản, (b) trợ lý giọng nói đọc to phản hồi,
(c) pipeline dịch tài liệu chạy ngầm ban đêm. Ứng dụng nào hưởng lợi nhiều
nhất từ streaming, ứng dụng nào không cần — và tại sao?** (1 đoạn văn)
> *Trong ba ứng dụng trên, chatbot văn bản hưởng lợi nhiều nhất từ streaming vì nó cải thiện trực tiếp trải nghiệm người dùng bằng cách giảm độ trễ cảm nhận, cho phép người dùng đọc từng phần phản hồi ngay lập tức thay vì phải chờ đợi toàn bộ văn bản sinh xong. Ngược lại, pipeline dịch tài liệu chạy ngầm ban đêm hoàn toàn không cần streaming vì đây là một tác vụ xử lý lô và không có sự tương tác thời gian thực với người dùng; việc nhận toàn bộ response một lần  giúp đơn giản hóa mã nguồn xử lý, dễ kiểm soát lỗi và tiết kiệm tài nguyên mạng. Với ứng dụng trợ lý giọng nói, streaming cũng cần thiết do khi kết hợp với công nghệ Streaming TTS để đọc thành tiếng ngay từ những từ đầu tiên, tránh khoảng lặng ngắt quãng*

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**Khi API quá tải và hàng nghìn client cùng retry, exponential backoff giúp
gì so với delay cố định? Tra cứu thêm: kỹ thuật "jitter" (thêm độ trễ ngẫu
nhiên) giải quyết vấn đề gì còn sót lại?**
> *Ý nghĩa của Exponential Backoff: Khi API bị quá tải và có hàng nghìn client cùng retry, việc tăng thời gian chờ theo cấp số nhân sau mỗi lần thất bại sẽ làm các lần retry diễn ra thưa dần, giúp giảm lượng request đổ về máy chủ trong cùng một thời điểm. Nhờ đó, server có thêm thời gian xử lý các yêu cầu đang tồn đọng và dần phục hồi. Trong khi đó,nếu dùng delay cố định, các client sẽ tiếp tục gửi lại request theo cùng một chu kỳ, khiến áp lực lên server giảm không đáng kể và tình trạng quá tải có thể kéo dài hơn

Vai trò của Jitter, do là chỉ dùng exponential backoff vẫn chưa đủ. Nếu nhiều client gặp lỗi gần như cùng lúc, vẫn có thể retry gần như đồng thời vì đều sử dụng cùng một khoảng thời gian backoff. Jitter khắc phục điều này bằng cách thêm một khoảng trễ ngẫu nhiên vào thời gian chờ của mỗi client, giúp các lần retry được phân tán theo thời gian thay vì dồn vào một thời điểm. Điều này làm giảm hiện tượng hàng loạt client cùng lúc gửi lại request và tiếp tục gây áp lực lên máy chủ*

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Viết lại system prompt bạn dùng cho trợ lý của mình. Chỉ ra 2 chỗ trong
prompt mà nếu xóa đi, hành vi trợ lý sẽ thay đổi rõ rệt — và mô tả thay đổi
đó:**
> *System Prompt của em: "Bạn là một trợ giảng AI thân thiện, am hiểu về lập trình Python và LLM API. Hãy trả lời ngắn gọn dưới 3 câu, luôn xưng 'mình' và gọi 'bạn', đồng thời tuyệt đối không đưa ra lời giải trực tiếp cho các bài tập về nhà mà chỉ gợi ý từng bước để người học tự giải."

Hai vị trí em thấy nếu xóa sẽ làm thay đổi rõ rệt hành vi là khi xóa yêu cầu "trả lời ngắn gọn dưới 3 câu" thì trợ lý sẽ không còn giới hạn độ dài câu trả lời, có xu hướng giải thích chi tiết hơn và tạo ra các phản hồi dài, làm giảm tính gọn gàng khi sử dụng trên giao diện CLI. Ngoài ra, khi xóa yêu cầu "không đưa ra lời giải trực tiếp cho bài tập về nhà, chỉ gợi ý từng bước",AI sẽ có xu hướng cung cấp luôn lời giải hoặc đoạn mã hoàn chỉnh khi được hỏi, thay vì đóng vai trò trợ giảng hỗ trợ người học tự suy luận*

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn giữ history 4 lượt cuối. Hãy mô tả một tình huống hội thoại
cụ thể mà giới hạn này khiến trợ lý trả lời sai/mất ngữ cảnh, và đề xuất một
cách khắc phục (ví dụ: tóm tắt các lượt cũ, tăng giới hạn có chọn lọc...):**
> *Tình huống giả sử người dùng đang cùng trợ lý xây dựng một chatbot bằng Python. Ở những lượt đầu, người dùng đã nói rằng dự án sử dụng FastAPI, OpenAI API và SQLite. Sau nhiều lượt trao đổi, các thông tin này bị đẩy ra ngoài phạm vi 4 lượt hội thoại gần nhất. Khi họ hỏi: "Hãy sửa tiếp phần API như lúc trước.", trợ lý không còn nhớ công nghệ và cấu trúc dự án đã thống nhất nên có thể đưa ra câu trả lời sai hoặc phải yêu cầu người dùng cung cấp lại thông tin. Cách khắc phục của em là tóm tắt các thông tin quan trọng của cuộc hội thoại (ví dụ: mục tiêu dự án, công nghệ sử dụng, các quyết định đã thống nhất và yêu cầu của người dùng). Đoạn tóm tắt này sẽ được cập nhật định kỳ và chèn ngay sau System Prompt ở mỗi lần gửi yêu cầu đến mô hình. Nhờ đó, dù chỉ lưu 4 lượt hội thoại gần nhất mà vẫn giữ được ngữ cảnh quan trọng xuyên suốt phiên chat mà không cần tăng đáng kể lượng history*

 đoạn tóm tắt này cố định vào ngay sau System Prompt để giữ ngữ cảnh xuyên suốt phiên chat.*
---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên GitHub cá nhân và nộp link repo vào vlearn (theo hướng dẫn README)
