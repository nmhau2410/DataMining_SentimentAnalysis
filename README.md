**Tên đề tài:**

Phân loại cảm xúc phản hồi tiếng Việt của sinh viên bằng Machine Learning, Convolutional Neural Network (CNN) và Transformer.

**Mô tả đề tài:**

**Bài toán:**  
Phân loại văn bản (Text Classification / Sentiment Analysis) nhằm tự động xác định cảm xúc của sinh viên dựa trên nội dung phản hồi bằng tiếng Việt. Mỗi phản hồi được phân loại thành một trong ba nhãn: **Tích cực (Positive), Tiêu cực (Negative), Trung tính (Neutral)**.

**Thuật toán và Mô hình:**  
- Multinomial Naive Bayes.
- Logistic Regression.
- Support Vector Machine (SVM).
- Convolutional Neural Network (CNN/TextCNN).
- Transformer sử dụng mô hình **PhoBERT** được huấn luyện trước cho tiếng Việt.

Các mô hình Machine Learning truyền thống sử dụng phương pháp biểu diễn văn bản **TF-IDF Vectorizer**. Mô hình CNN sử dụng Word Embedding để học đặc trưng văn bản, trong khi PhoBERT được fine-tuning trực tiếp trên tập dữ liệu để khai thác thông tin ngữ cảnh của câu.

Mục tiêu là xây dựng và so sánh hiệu quả của ba hướng tiếp cận: **Machine Learning truyền thống, Deep Learning sử dụng CNN và Transformer** trong bài toán phân loại cảm xúc văn bản tiếng Việt.

**Tập dữ liệu:**  
**UIT-VSFC (Vietnamese Students’ Feedback Corpus)** – bộ dữ liệu phản hồi của sinh viên bằng tiếng Việt do nhóm nghiên cứu NLP thuộc Trường Đại học Công nghệ Thông tin – Đại học Quốc gia TP.HCM xây dựng và công bố.

Tập dữ liệu gồm hơn **16.000 câu phản hồi tiếng Việt được gán nhãn thủ công**, phục vụ hai bài toán là phân loại cảm xúc và phân loại chủ đề. Trong đề tài này, nhóm chỉ sử dụng nhãn cảm xúc gồm:

- Negative – Tiêu cực.
- Neutral – Trung tính.
- Positive – Tích cực.

**Link tập dữ liệu:**  
UIT NLP Group – UIT-VSFC Dataset (nguồn chính thức của Trường Đại học Công nghệ Thông tin – ĐHQG TP.HCM).

**Bài báo nghiên cứu khoa học của tập dữ liệu:**  
Kiet Van Nguyen, Vu Duc Nguyen, Phu Xuan-Vinh Nguyen, Tham Thi-Hong Truong, Ngan Luu-Thuy Nguyen, **“UIT-VSFC: Vietnamese Students’ Feedback Corpus for Sentiment Analysis”**, 2018 10th International Conference on Knowledge and Systems Engineering (KSE 2018), IEEE, pp. 19–24. DOI: **10.1109/KSE.2018.8573337**.

**Quy trình tiền xử lý:**

- Kiểm tra và xử lý dữ liệu khuyết hoặc dữ liệu không hợp lệ.
- Loại bỏ các phản hồi trùng lặp nếu có.
- Chuẩn hóa văn bản tiếng Việt như khoảng trắng, ký tự đặc biệt và các ký tự không cần thiết.
- Kiểm tra phân bố ba lớp Positive, Negative và Neutral để phát hiện tình trạng mất cân bằng dữ liệu.
- Sử dụng các tập Train, Validation/Development và Test để huấn luyện và đánh giá mô hình.

**Đối với Machine Learning:**
- Chuyển văn bản về dạng chuẩn.
- Tách từ tiếng Việt.
- Có thể loại bỏ Stopwords đối với các thử nghiệm cần thiết.
- Chuyển văn bản thành vector số bằng **TF-IDF Vectorizer**.
- Huấn luyện Multinomial Naive Bayes, Logistic Regression và SVM.

**Đối với CNN:**
- Chuẩn hóa và tách từ văn bản.
- Tokenization để chuyển văn bản thành chuỗi token.
- Chuyển token thành chỉ số.
- Padding để đưa các câu về cùng độ dài.
- Sử dụng Embedding Layer kết hợp **Conv1D và Global Max Pooling** để trích xuất đặc trưng văn bản.
- Sử dụng lớp Softmax để phân loại thành ba nhãn cảm xúc.

**Đối với Transformer – PhoBERT:**
- Chỉ thực hiện chuẩn hóa văn bản ở mức cần thiết, không loại bỏ quá nhiều thông tin của câu.
- Thực hiện tách từ tiếng Việt.
- Tokenization bằng tokenizer tương ứng của PhoBERT.
- Padding và Truncation để chuẩn hóa độ dài đầu vào.
- Fine-tuning **PhoBERT** cho bài toán phân loại ba lớp Positive, Negative và Neutral.

**Xử lý mất cân bằng dữ liệu:**  
Kiểm tra sự chênh lệch số lượng giữa các lớp cảm xúc. Trong trường hợp cần thiết, sử dụng **Class Weight / Weighted Loss** cho các mô hình hỗ trợ nhằm giảm ảnh hưởng của lớp có số lượng mẫu lớn và cải thiện khả năng nhận diện lớp có ít dữ liệu.

**Ứng dụng demo:**  
Xây dựng giao diện Web phân tích cảm xúc văn bản tiếng Việt bằng **Streamlit hoặc Gradio**.

**Chức năng:**
- Cho phép người dùng nhập trực tiếp một câu phản hồi bằng tiếng Việt.
- Sử dụng mô hình có hiệu suất tốt nhất để tự động dự đoán phản hồi thuộc nhóm **Tích cực, Tiêu cực hoặc Trung tính**.
- Hiển thị độ tin cậy/xác suất của kết quả dự đoán.
- Hiển thị biểu đồ phân bố các lớp cảm xúc trong tập dữ liệu.
- Hiển thị WordCloud hoặc biểu đồ các từ xuất hiện nổi bật trong từng nhóm cảm xúc.
- Hiển thị bảng hoặc biểu đồ so sánh hiệu suất của các mô hình Machine Learning, CNN và Transformer.

**Đơn vị đo hiệu suất mô hình:**
- Accuracy.
- Precision.
- Recall.
- Macro F1-Score.
- Confusion Matrix.

Trong đó, **Macro F1-Score** được ưu tiên khi so sánh các mô hình vì tập dữ liệu có sự chênh lệch về số lượng mẫu giữa các lớp cảm xúc.
