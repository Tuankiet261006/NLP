# Báo cáo so sánh dịch máy Anh - Việt

## Kết quả định lượng

| Phương pháp | BLEU | chrF | Câu | Thời gian train (s) | Thời gian dịch (s) | Câu/s | Tham số | Thiết bị |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| transformer | 0.4185 | 8.5468 | 1268 | 1281.5379 | 549.1909 | 2.3089 | 75195826 | cuda (NVIDIA GeForce RTX 3060 Laptop GPU) |
| tensorflow_nmt_tf2 | 1.4242 | 10.9471 | 1268 | N/A | 3679.5526 | 0.3446 | 63469483 | /physical_device:GPU:0 |

Các số thời gian chỉ được so sánh trực tiếp khi hai lần chạy dùng phần cứng tương đương.

## Chữ ký metric

- Transformer BLEU: nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.5.1
- Transformer chrF: nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.5.1
- TensorFlow NMT BLEU: nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.5.1
- TensorFlow NMT chrF: nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.5.1

## Phân tích định tính

| # | Nhóm | English | Tham chiếu | Transformer | TensorFlow NMT |
|---:|---|---|---|---|---|
| 120 | Câu ngắn | Extraordinary . | Phi thường . | Bạn có thể làm gì ? | <unk> |
| 779 | Câu ngắn | Absolutely . | Chính xác . | Đó là một điều tuyệt vời . | Chắc chắn . |
| 917 | Câu ngắn | Thanks . | Cảm ơn | Xin cảm ơn . | Cảm ơn . |
| 221 | Câu dài | Instead , Conor came home one Friday evening and he told me that he had quit his job that day , his dream job , and he said that he had quit his job because of me , because I had made him feel so safe and loved that he didn &apos;t need to prove himself on Wall Street anymore , and he just wanted to get out of the city and away from his abusive , dysfunctional family , and move to a tiny town in New England where he could start his life over with me by his side . | Thay vì vậy , Conor đã về nhà vào một tối thứ sáu và anh ấy nói với tôi rằng anh ấy nghỉ việc ngày hôm nay công việc mơ ước của anh ta , và anh ta nói rằng , anh ta bỏ việc vì tôi , bởi vì tôi làm anh cảm thấy thật an toàn và được yêu thương rằng anh ta không cần chứng tỏ bản thân ở phố Wall thêm nửa và anh ta chỉ muốn đi khỏi thành phố này xa rời gia đình bất thường và đầy ngược đãi của anh ta , đi đến một thị trấn nhỏ ở New England nơi anh ta có thể bắt đầu lại cuộc sống của anh với tôi | Và tôi nói , &quot; Chúng ta có thể làm điều đó . &quot; | Thay <unk> |
| 643 | Câu dài | So if the photographer is right there and the light is right there , like a nice HMI , and the client says , &quot; Cameron , we want a walking shot , &quot; well then this leg goes first , nice and long , this arm goes back , this arm goes forward , the head is at three quarters , and you just go back and forth , just do that , and then you look back at your imaginary friends , 300 , 400 , 500 times . | Nếu nhiếp ảnh gia đứng ở đây còn ánh sáng thì ở kia , giống như là một giao diện tương tác đẹp , và khi khách hàng bảo : &quot; Cameron chúng tôi muốn bức ảnh em đang bước đi &quot; , thế là cái chân này đi trước , đẹp và dài , cái tay này thì đánh về sau , còn cái này thì đưa ra trước , đầu thì ở vị trí ba phần tư , và bạn chỉ đơn giản là đi tới đi lui , chỉ cần làm điều đó , và sau đó bạn quay đầu nhìn phía sau , tưởng tượng đó là bạn bè của bạn , 300 , 400 , 500 lần . | Và tôi nghĩ , tôi sẽ nói với các bạn rằng chúng ta có thể làm điều gì đó , và tôi sẽ nói với bạn rằng chúng ta có thể làm điều đó . | Vì vậy , nếu tổ chức viên <unk> |
| 1008 | Câu dài | And so what I did in Esperance that first year was to just walk the streets , and in three days I had my first client , and I helped this first guy who was smoking fish from a garage , was a Maori guy , and I helped him to sell to the restaurant in Perth , to get organized , and then the fishermen came to me to say , &quot; You the guy who helped Maori ? Can you help us ? &quot; | Vậy nên cái mà tôi làm trong năm đầu tiên tại Esperance là chỉ bách bộ trên đường , và trong 3 ngày Tôi có khách hàng đầu tiên Một người làm nghề hun khói cá trong nhà xe , tên là Maori , và tôi giúp anh ta bán cá cho một nhà hàng tại Perth , và tổ chức lại công việc , sau đó thì một người đánh cá tìm đến tôi và nói Ông là người đã giúp Maori hả ? Ông có thể giúp tôi với không ? | Và tôi nói , &quot; Tôi không thể làm được điều đó . &quot; | <unk> |
| 1 | Mẫu phân bố | When I was little , I thought my country was the best on the planet , and I grew up singing a song called &quot; Nothing To Envy . &quot; | Khi tôi còn nhỏ , Tôi nghĩ rằng BắcTriều Tiên là đất nước tốt nhất trên thế giới và tôi thường hát bài &quot; Chúng ta chẳng có gì phải ghen tị . &quot; | Và tôi nói , &quot; Tôi có thể làm việc với những người phụ nữ này . &quot; | Khi tôi còn nhỏ , tôi nghĩ tôi đã <unk> |
| 318 | Mẫu phân bố | We have eight species of vultures that occur in Kenya , of which six are highly threatened with extinction . | Chúng ta có 8 loài kền kền xuất hiện tại Kenya , trong đó có 6 loài bị đe doạ với nguy cơ tuyệt chủng cao . | Chúng ta có thể làm điều đó . | Chúng tôi có 8 <unk> |
| 635 | Mẫu phân bố | But the second answer , and what I really want to say to these little girls is , &quot; Why ? | Nhưng câu trả lời tiếp theo mà tôi thực sự muốn nói với những bé gái đó là : &quot; Tại sao chứ ? &quot; | Và tôi nói , &quot; Được rồi . &quot; | Nhưng câu trả lời thứ hai , <unk> |
| 952 | Mẫu phân bố | So we arrived there with Italian seeds in southern Zambia in this absolutely magnificent valley going down to the Zambezi River , and we taught the local people how to grow Italian tomatoes and zucchini and ... | Thế nên chúng tôi miền nam Zambia với những hạt giống từ nước Ý tại một thung lũng tuyệt đẹp trải dài xuống sông Zambezi , và chúng tôi dạy cho người dân bản đại trồng cà chua Ý và bí ngòi Ý và ... | Chúng ta có thể làm điều đó . | <unk> |

Cần bổ sung nhận xét thủ công về sai nghĩa, từ hiếm, trật tự từ và mức tự nhiên của từng ví dụ trước khi chốt báo cáo.

## Kết luận

Điền kết luận sau khi kiểm tra metric, điều kiện phần cứng và các ví dụ dịch định tính.
