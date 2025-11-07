---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:12186
- loss:CosineSimilarityLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: Doanh nghiệp tư nhân có được giảm vốn đầu tư xuống thấp hơn số
    vốn đầu tư đã đăng ký hay không?
  sentences:
  - 'Dấu của doanh nghiệp được quy định tại Điều 43 Luật Doanh nghiệp 2020 như sau:

    Dấu của doanh nghiệp

    1. Dấu bao gồm dấu được làm tại cơ sở khắc dấu hoặc dấu dưới hình thức chữ ký
    số theo quy định của pháp luật về giao dịch điện tử.

    2. Doanh nghiệp quyết định loại dấu, số lượng, hình thức và nội dung dấu của doanh
    nghiệp, chi nhánh, văn phòng đại diện và đơn vị khác của doanh nghiệp.

    3. Việc quản lý và lưu giữ dấu thực hiện theo quy định của Điều lệ công ty hoặc
    quy chế do doanh nghiệp, chi nhánh, văn phòng đại diện hoặc đơn vị khác của doanh
    nghiệp có dấu ban hành. Doanh nghiệp sử dụng dấu trong các giao dịch theo quy
    định của pháp luật.

    Theo quy định nêu trên doanh nghiệp quyết định loại dấu, số lượng, hình thức và
    nội dung dấu của doanh nghiệp, chi nhánh, văn phòng đại diện và đơn vị khác của
    doanh nghiệp.

    Theo đó, công ty cổ phần được tự quyền quyết định mẫu con dấu, số lượng, hình
    thức và nội dung dấu của công ty mình, pháp luật không có quy định về số lượng
    con dấu tối đa. Cho nên, công ty cổ phần có thể sử dụng đồng thời 2 con dấu.

    '
  - 'Theo quy định tại khoản 3 Điều 163 Luật Doanh nghiệp 2020, lương và thưởng của
    Giám đốc công ty được tính vào chi phí kinh doanh của doanh nghiệp theo quy định
    của pháp luật về thuế thu nhập doanh nghiệp như sau:

    "Điều 163. Tiền lương, thù lao, thưởng và lợi ích khác của thành viên Hội đồng
    quản trị, Giám đốc, Tổng Giám đốc

    1. Công ty có quyền trả thù lao, thưởng cho thành viên Hội đồng quản trị, trả
    lương, thưởng cho Giám đốc hoặc Tổng giám đốc và người quản lý khác theo kết quả
    và hiệu quả kinh doanh.

    ...

    3. Thù lao của từng thành viên Hội đồng quản trị, tiền lương của Giám đốc hoặc
    Tổng giám đốc và người quản lý khác được tính vào chi phí kinh doanh của công
    ty theo quy định của pháp luật về thuế thu nhập doanh nghiệp, được thể hiện thành
    mục riêng trong báo cáo tài chính hằng năm của công ty và phải báo cáo Đại hội
    đồng cổ đông tại cuộc họp thường niên."

    Như vậy, để tiền lương và thưởng của công ty chi trả cho giám đốc công ty được
    tính vào chi phí được trừ của doanh nghiệp khi tính thuế TNDN thì phải đáp ứng
    các điều kiện sau đây:

    - Doanh nghiệp phải có chi trả thực tế và có chứng từ thanh toán tiền lương, thưởng
    cho Giám đốc.

    - Các khoản tiền lương, tiền thưởng cho Giám đốc của doanh nghiệp phải được ghi
    cụ thể về điều kiện hưởng và mức hưởng tại tất cả các tài liệu sau: Hợp đồng lao
    động; Thoả ước lao động tập thể; Quy chế tài chính của Công ty; Quy chế thưởng
    do Chủ tịch Hội đồng quản trị, Giám đốc quy định theo quy chế tài chính của Công
    ty.

    '
  - 'Căn cứ tại khoản 2 Điều 14 Luật Doanh nghiệp 2020 có quy định như sau:

    Người đại diện theo ủy quyền của chủ sở hữu, thành viên, cổ đông công ty là tổ
    chức

    1. Người đại diện theo ủy quyền của chủ sở hữu, thành viên, cổ đông công ty là
    tổ chức phải là cá nhân được ủy quyền bằng văn bản nhân danh chủ sở hữu, thành
    viên, cổ đông đó thực hiện quyền và nghĩa vụ theo quy định của Luật này.

    2. Trường hợp Điều lệ công ty không có quy định khác thì việc cử người đại diện
    theo ủy quyền thực hiện theo quy định sau đây:

    a) Tổ chức là thành viên công ty trách nhiệm hữu hạn hai thành viên trở lên có
    sở hữu ít nhất 35% vốn điều lệ có thể ủy quyền tối đa 03 người đại diện theo ủy
    quyền;

    b) Tổ chức là cổ đông công ty cổ phần có sở hữu ít nhất 10% tổng số cổ phần phổ
    thông có thể ủy quyền tối đa 03 người đại diện theo ủy quyền.

    3. Trường hợp chủ sở hữu, thành viên, cổ đông công ty là tổ chức cử nhiều người
    đại diện theo ủy quyền thì phải xác định cụ thể phần vốn góp, số cổ phần cho mỗi
    người đại diện theo ủy quyền. Trường hợp chủ sở hữu, thành viên, cổ đông công
    ty không xác định phần vốn góp, số cổ phần tương ứng cho mỗi người đại diện theo
    ủy quyền thì phần vốn góp, số cổ phần sẽ được chia đều cho tất cả người đại diện
    theo ủy quyền.

    …

    Như vậy, theo quy định trên thì trừ trường hợp Điều lệ công ty không có quy định
    khác thì tổ chức làm chủ sở hữu doanh nghiệp được người đại diện theo ủy quyền
    như sau:

    - Tổ chức là thành viên công ty trách nhiệm hữu hạn hai thành viên trở lên có
    sở hữu ít nhất 35% vốn điều lệ có thể ủy quyền tối đa 03 người đại diện theo ủy
    quyền;

    - Tổ chức là cổ đông công ty cổ phần có sở hữu ít nhất 10% tổng số cổ phần phổ
    thông có thể ủy quyền tối đa 03 người đại diện theo ủy quyền.

    '
- source_sentence: Việc góp vốn khi thành lập doanh nghiệp chỉ được coi là thanh toán
    xong khi nào?
  sentences:
  - "Căn cứ tại khoản 3 Điều 4 Luật Doanh nghiệp 2020 thì cổ đông là cá nhân, tổ chức\
    \ sở hữu ít nhất một cổ phần của công ty cổ phần.\nCổ đông là người góp vốn đóng\
    \ góp vào cho công ty, nắm giữ sở hữu tỉ lệ cổ phần tương đương với phần vốn góp\
    \ đã được góp vào công ty. \n"
  - 'Căn cứ tại Điều 7 Luật Doanh nghiệp 2020 thì công ty trách nhiệm hữu hạn một
    thành viên có quyền sau:

    - Tự do kinh doanh ngành, nghề mà luật không cấm.

    - Tự chủ kinh doanh và lựa chọn hình thức tổ chức kinh doanh; chủ động lựa chọn
    ngành, nghề, địa bàn, hình thức kinh doanh; chủ động điều chỉnh quy mô và ngành,
    nghề kinh doanh.

    - Lựa chọn hình thức, phương thức huy động, phân bổ và sử dụng vốn.

    - Tự do tìm kiếm thị trường, khách hàng và ký kết hợp đồng.

    - Kinh doanh xuất khẩu, nhập khẩu.

    - Tuyển dụng, thuê và sử dụng lao động theo quy định của pháp luật về lao động.

    - Chủ động ứng dụng khoa học và công nghệ để nâng cao hiệu quả kinh doanh và khả
    năng cạnh tranh; được bảo hộ quyền sở hữu trí tuệ theo quy định của pháp luật
    về sở hữu trí tuệ.

    - Chiếm hữu, sử dụng, định đoạt tài sản của doanh nghiệp.

    - Từ chối yêu cầu của cơ quan, tổ chức, cá nhân về cung cấp nguồn lực không theo
    quy định của pháp luật.

    - Khiếu nại, tham gia tố tụng theo quy định của pháp luật.

    - Quyền khác theo quy định của pháp luật.

    Căn cứ tại Điều 8 Luật Doanh nghiệp 2020 thì công ty trách nhiệm hữu hạn một thành
    viên có nghĩa vụ như sau:

    - Đáp ứng đủ điều kiện đầu tư kinh doanh khi kinh doanh ngành, nghề đầu tư kinh
    doanh có điều kiện; ngành, nghề tiếp cận thị trường có điều kiện đối với nhà đầu
    tư nước ngoài theo quy định của pháp luật và bảo đảm duy trì đủ điều kiện đó trong
    suốt quá trình hoạt động kinh doanh.

    - Thực hiện đầy đủ, kịp thời nghĩa vụ về đăng ký doanh nghiệp, đăng ký thay đổi
    nội dung đăng ký doanh nghiệp, công khai thông tin về thành lập và hoạt động của
    doanh nghiệp, báo cáo và nghĩa vụ khác theo quy định của Luật này.

    - Chịu trách nhiệm về tính trung thực, chính xác của thông tin kê khai trong hồ
    sơ đăng ký doanh nghiệp và các báo cáo; trường hợp phát hiện thông tin đã kê khai
    hoặc báo cáo thiếu chính xác, chưa đầy đủ thì phải kịp thời sửa đổi, bổ sung các
    thông tin đó.

    - Tổ chức công tác kế toán, nộp thuế và thực hiện các nghĩa vụ tài chính khác
    theo quy định của pháp luật.

    - Bảo đảm quyền, lợi ích hợp pháp, chính đáng của người lao động theo quy định
    của pháp luật; không phân biệt đối xử, xúc phạm danh dự, nhân phẩm của người lao
    động trong doanh nghiệp; không ngược đãi lao động, cưỡng bức lao động hoặc sử
    dụng lao động chưa thành niên trái pháp luật; hỗ trợ và tạo điều kiện thuận lợi
    cho người lao động tham gia đào tạo nâng cao trình độ, kỹ năng nghề; thực hiện
    các chính sách, chế độ bảo hiểm xã hội, bảo hiểm thất nghiệp, bảo hiểm y tế và
    bảo hiểm khác cho người lao động theo quy định của pháp luật.

    - Nghĩa vụ khác theo quy định của pháp luật.

    '
  - 'Theo quy định tại khoản 1 Điều 8 Nghị định 153/2020/NĐ-CP (được sửa đổi bởi khoản
    6 Điều 1 Nghị định 65/2022/NĐ-CP) quy định về nhà đầu tư trái phiếu như sau:

    Nhà đầu tư mua trái phiếu

    1. Đối tượng mua trái phiếu

    a) Đối với trái phiếu không chuyển đổi không kèm chứng quyền: đối tượng mua trái
    phiếu là nhà đầu tư chứng khoán chuyên nghiệp theo quy định của pháp luật chứng
    khoán.

    b) Đối với trái phiếu chuyển đổi và trái phiếu kèm chứng quyền: đối tượng mua
    trái phiếu là nhà đầu tư chứng khoán chuyên nghiệp, nhà đầu tư chiến lược, trong
    đó số lượng nhà đầu tư chiến lược phải đảm bảo dưới 100 nhà đầu tư.

    c) Nhà đầu tư chứng khoán chuyên nghiệp là nhà đầu tư có năng lực tài chính hoặc
    có trình độ chuyên môn về chứng khoán theo quy định tại Điều 11 Luật Chứng khoán.
    Tổ chức có trách nhiệm xác định nhà đầu tư chứng khoán chuyên nghiệp và tài liệu
    xác định nhà đầu tư chứng khoán chuyên nghiệp thực hiện theo quy định tại Điều
    4 và Điều 5 Nghị định số 155/2020/NĐ-CP ngày 31 tháng 12 năm 2020 của Chính phủ
    quy định chi tiết thi hành một số điều của Luật Chứng khoán và các văn bản sửa
    đổi, bổ sung, thay thế (sau đây gọi tắt là Nghị định số 155/2020/NĐ-CP), ngoại
    trừ việc xác định nhà đầu tư chứng khoán chuyên nghiệp quy định tại điểm d khoản
    này.

    d) Việc xác định nhà đầu tư chứng khoán chuyên nghiệp là cá nhân theo quy định
    tại điểm d khoản 1 Điều 11 Luật Chứng khoán để mua trái phiếu doanh nghiệp phát
    hành riêng lẻ phải đảm bảo danh mục chứng khoán niêm yết, đăng ký giao dịch do
    nhà đầu tư nắm giữ có giá trị tối thiểu 02 tỷ đồng được xác định bằng giá trị
    thị trường bình quân theo ngày của danh mục chứng khoán trong thời gian tối thiểu
    180 ngày liền kề trước ngày xác định tư cách nhà đầu tư chứng khoán chuyên nghiệp,
    không bao gồm giá trị vay giao dịch ký quỹ và giá trị chứng khoán thực hiện giao
    dịch mua bán lại. Việc xác định nhà đầu tư chứng khoán chuyên nghiệp tại điểm
    này có giá trị trong vòng 03 tháng kể từ ngày được xác nhận.

    Như vậy, theo quy định thì nhà đầu tư mua trái phiếu phải là nhà đầu tư chứng
    khoán chuyên nghiệp; nhà đầu tư chiến lược và phải bảo đảm danh mục nắm giữ có
    giá trị trung bình từ 2 tỷ đồng tối thiểu trong vòng 180 ngày bằng tài sản của
    nhà đầu tư, không bao gồm tiền vay.

    Đối với quy định này, các chuyên gia cho rằng hiện nay, có nhiều nhà đầu tư dưới
    chuẩn đang mua trái phiếu doanh nghiệp riêng lẻ. Nếu thực hiện theo quy định thì
    họ sẽ không được xác định là nhà đầu tư chứng khoán chuyên nghiệp.

    '
- source_sentence: Ai là người có thẩm quyền tổ chức kiểm phiếu theo quy định?
  sentences:
  - 'Căn cứ theo quy định tại Điều 10 Luật Doanh nghiệp 2020 có quy định về tiêu chí,
    quyền và nghĩa vụ của doanh nghiệp xã hội như sau:

    Tiêu chí, quyền và nghĩa vụ của doanh nghiệp xã hội

    1. Doanh nghiệp xã hội phải đáp ứng các tiêu chí sau đây:

    a) Là doanh nghiệp được đăng ký thành lập theo quy định của Luật này;

    b) Mục tiêu hoạt động nhằm giải quyết vấn đề xã hội, môi trường vì lợi ích cộng
    đồng;

    c) Sử dụng ít nhất 51% tổng lợi nhuận sau thuế hằng năm của doanh nghiệp để tái
    đầu tư nhằm thực hiện mục tiêu đã đăng ký.

    2. Ngoài quyền và nghĩa vụ của doanh nghiệp theo quy định của Luật này, doanh
    nghiệp xã hội có quyền và nghĩa vụ sau đây:

    a) Chủ sở hữu, người quản lý doanh nghiệp xã hội được xem xét, tạo thuận lợi và
    hỗ trợ trong việc cấp giấy phép, chứng chỉ và giấy chứng nhận có liên quan theo
    quy định của pháp luật;

    b) Được huy động, nhận tài trợ từ cá nhân, doanh nghiệp, tổ chức phi chính phủ
    và tổ chức khác của Việt Nam, nước ngoài để bù đắp chi phí quản lý, chi phí hoạt
    động của doanh nghiệp;

    c) Duy trì mục tiêu hoạt động và điều kiện quy định tại điểm b và điểm c khoản
    1 Điều này trong suốt quá trình hoạt động;

    d) Không được sử dụng các khoản tài trợ huy động được cho mục đích khác ngoài
    bù đắp chi phí quản lý và chi phí hoạt động để giải quyết vấn đề xã hội, môi trường
    mà doanh nghiệp đã đăng ký;

    đ) Trường hợp được nhận các ưu đãi, hỗ trợ, doanh nghiệp xã hội phải định kỳ hằng
    năm báo cáo cơ quan có thẩm quyền về tình hình hoạt động của doanh nghiệp.

    3. Doanh nghiệp xã hội phải thông báo với cơ quan có thẩm quyền khi chấm dứt thực
    hiện mục tiêu xã hội, môi trường hoặc không sử dụng lợi nhuận để tái đầu tư theo
    quy định tại điểm b và điểm c khoản 1 Điều này.

    4. Nhà nước có chính sách khuyến khích, hỗ trợ và thúc đẩy phát triển doanh nghiệp
    xã hội.

    5. Chính phủ quy định chi tiết Điều này.

    Theo quy định về các tiêu chí mà doanh nghiệp xã hội phải đáp ứng thì doanh nghiệp
    xã hội phải sử dụng ít nhất 51% tổng lợi nhuận sau thuế hằng năm của doanh nghiệp
    để tái đầu tư nhằm thực hiện mục tiêu đã đăng ký.

    Như vậy, doanh nghiệp xã hội có thể sử dụng 100% tổng lợi nhuận sau thuế hằng
    năm để tái đầu tư để thực hiện mục tiêu đã đăng ký là hoạt động nhằm giải quyết
    vấn đề xã hội, môi trường vì lợi ích cộng đồng.

    '
  - 'Theo quy định tại Điều 112 Luật Doanh nghiệp 2020 quy định về trường hợp thực
    hiện giảm vốn điều lệ công ty cụ thể như sau:

    Vốn của công ty cổ phần

    1. Vốn điều lệ của công ty cổ phần là tổng mệnh giá cổ phần các loại đã bán. Vốn
    điều lệ của công ty cổ phần khi đăng ký thành lập doanh nghiệp là tổng mệnh giá
    cổ phần các loại đã được đăng ký mua và được ghi trong Điều lệ công ty.

    2. Cổ phần đã bán là cổ phần được quyền chào bán đã được các cổ đông thanh toán
    đủ cho công ty. Khi đăng ký thành lập doanh nghiệp, cổ phần đã bán là tổng số
    cổ phần các loại đã được đăng ký mua.

    3. Cổ phần được quyền chào bán của công ty cổ phần là tổng số cổ phần các loại
    mà Đại hội đồng cổ đông quyết định sẽ chào bán để huy động vốn. Số cổ phần được
    quyền chào bán của công ty cổ phần khi đăng ký thành lập doanh nghiệp là tổng
    số cổ phần các loại mà công ty sẽ chào bán để huy động vốn, bao gồm cổ phần đã
    được đăng ký mua và cổ phần chưa được đăng ký mua.

    4. Cổ phần chưa bán là cổ phần được quyền chào bán và chưa được thanh toán cho
    công ty. Khi đăng ký thành lập doanh nghiệp, cổ phần chưa bán là tổng số cổ phần
    các loại chưa được đăng ký mua.

    5. Công ty có thể giảm vốn điều lệ trong trường hợp sau đây:

    a) Theo quyết định của Đại hội đồng cổ đông, công ty hoàn trả một phần vốn góp
    cho cổ đông theo tỷ lệ sở hữu cổ phần của họ trong công ty nếu công ty đã hoạt
    động kinh doanh liên tục từ 02 năm trở lên kể từ ngày đăng ký thành lập doanh
    nghiệp và bảo đảm thanh toán đủ các khoản nợ và nghĩa vụ tài sản khác sau khi
    đã hoàn trả cho cổ đông;

    b) Công ty mua lại cổ phần đã bán theo quy định tại Điều 132 và Điều 133 của Luật
    này;

    c) Vốn điều lệ không được các cổ đông thanh toán đầy đủ và đúng hạn theo quy định
    tại Điều 113 của Luật này.

    Căn cứ vào quy định trên thì công ty cổ phần có thể giảm vốn điều lệ trong các
    trường hợp sau đây:

    - Theo quyết định của Đại hội đồng cổ đông, công ty hoàn trả một phần vốn góp
    cho cổ đông theo tỷ lệ sở hữu cổ phần của họ trong công ty nếu công ty đã hoạt
    động kinh doanh liên tục từ 02 năm trở lên kể từ ngày đăng ký thành lập doanh
    nghiệp và bảo đảm thanh toán đủ các khoản nợ và nghĩa vụ tài sản khác sau khi
    đã hoàn trả cho cổ đông;

    - Công ty mua lại cổ phần đã bán theo quy định tại Điều 132 và Điều 133 của Luật
    này;

    - Vốn điều lệ không được các cổ đông thanh toán đầy đủ và đúng hạn theo quy định
    tại Điều 113 của Luật này.

    '
  - 'Theo quy định tại Điều 11 Luật Doanh nghiệp 2020 về chế độ lưu giữ tài liệu của
    doanh nghiệp như sau:

    “Điều 11. Chế độ lưu giữ tài liệu của doanh nghiệp

    1. Tùy theo loại hình, doanh nghiệp phải lưu giữ các tài liệu sau đây:

    a) Điều lệ công ty; quy chế quản lý nội bộ của công ty; sổ đăng ký thành viên
    hoặc sổ đăng ký cổ đông;

    b) Văn bằng bảo hộ quyền sở hữu công nghiệp; giấy chứng nhận đăng ký chất lượng
    sản phẩm, hàng hóa, dịch vụ; giấy phép và giấy chứng nhận khác;

    c) Tài liệu, giấy tờ xác nhận quyền sở hữu tài sản của công ty;

    d) Phiếu biểu quyết, biên bản kiểm phiếu, biên bản họp Hội đồng thành viên, Đại
    hội đồng cổ đông, Hội đồng quản trị; các quyết định của doanh nghiệp;

    đ) Bản cáo bạch để chào bán hoặc niêm yết chứng khoán;

    e) Báo cáo của Ban kiểm soát, kết luận của cơ quan thanh tra, kết luận của tổ
    chức kiểm toán;

    g) Sổ kế toán, chứng từ kế toán, báo cáo tài chính hằng năm.

    2. Doanh nghiệp phải lưu giữ các tài liệu quy định tại khoản 1 Điều này tại trụ
    sở chính hoặc địa điểm khác được quy định trong Điều lệ công ty; thời hạn lưu
    giữ thực hiện theo quy định của pháp luật.”

    Thời hạn lưu giữ Điều lệ Công ty trách nhiệm hữu hạn được căn cứ vào Bảng thời
    hạn bảo quản hồ sơ, tài liệu hình thành phổ biến trong hoạt động của các cơ quan,
    tổ chức (ban hành kèm theo Thông tư 09/2011/TT-BNV) như sau:

    Tuy nhiên, Thông tư 09/2011/TT-BNV đã hết hiệu lực được thay thế bởi Thông tư
    10/2022/TT-BNV, theo mục 3 Phụ lục I kèm theo Thông tư 10/2022/TT-BNV quy định
    về Tài liệu tổ chức, cán bộ, công chức, viên chức, người lao động như sau:

    Hồ sơ xây dựng, ban hành Điều lệ tổ chức và hoạt động của cơ quan, tổ chức; Quy
    chế làm việc, quy định, hướng dẫn công tác tổ chức của cơ quan, tổ chức có thời
    hạn bảo quản 20 năm.

    Căn cứ theo các quy định pháp luật nêu trên, Điều lệ Công ty trách nhiệm hữu hạn
    phải được lưu giữ tại trụ sở chính hoặc địa điểm khác được quy định trong Điều
    lệ công ty với thời hạn lưu giữ 20 năm chứ không còn thời hạn bảo quản vĩnh viễn
    như quy định cũ.

    Tải về Tổng hợp mẫu Điều lệ Công ty TNHH mới nhất 2024

    '
- source_sentence: Hồ sơ đăng ký doanh nghiệp của công ty TNHH qua mạng thông tin
    điện tử có giá trị pháp lý như hồ sơ giấy không?
  sentences:
  - 'Hồ sơ đăng ký doanh nghiệp của công ty TNHH qua mạng thông tin điện tử được quy
    định tại Điều 26 Luật Doanh nghiệp 2020 như sau:

    Trình tự, thủ tục đăng ký doanh nghiệp

    1. Người thành lập doanh nghiệp hoặc người được ủy quyền thực hiện đăng ký doanh
    nghiệp với Cơ quan đăng ký kinh doanh theo phương thức sau đây:

    a) Đăng ký doanh nghiệp trực tiếp tại Cơ quan đăng ký kinh doanh;

    b) Đăng ký doanh nghiệp qua dịch vụ bưu chính;

    c) Đăng ký doanh nghiệp qua mạng thông tin điện tử.

    2. Đăng ký doanh nghiệp qua mạng thông tin điện tử là việc người thành lập doanh
    nghiệp nộp hồ sơ đăng ký doanh nghiệp qua mạng thông tin điện tử tại Cổng thông
    tin quốc gia về đăng ký doanh nghiệp. Hồ sơ đăng ký doanh nghiệp qua mạng thông
    tin điện tử bao gồm các dữ liệu theo quy định của Luật này và được thể hiện dưới
    dạng văn bản điện tử. Hồ sơ đăng ký doanh nghiệp qua mạng thông tin điện tử có
    giá trị pháp lý tương đương hồ sơ đăng ký doanh nghiệp bằng bản giấy.

    3. Tổ chức, cá nhân có quyền lựa chọn sử dụng chữ ký số theo quy định của pháp
    luật về giao dịch điện tử hoặc sử dụng tài khoản đăng ký kinh doanh để đăng ký
    doanh nghiệp qua mạng thông tin điện tử.

    4. Tài khoản đăng ký kinh doanh là tài khoản được tạo bởi Hệ thống thông tin quốc
    gia về đăng ký doanh nghiệp, cấp cho cá nhân để thực hiện đăng ký doanh nghiệp
    qua mạng thông tin điện tử. Cá nhân được cấp tài khoản đăng ký kinh doanh chịu
    trách nhiệm trước pháp luật về việc đăng ký để được cấp và việc sử dụng tài khoản
    đăng ký kinh doanh để đăng ký doanh nghiệp qua mạng thông tin điện tử.

    5. Trong thời hạn 03 ngày làm việc kể từ ngày nhận hồ sơ, Cơ quan đăng ký kinh
    doanh có trách nhiệm xem xét tính hợp lệ của hồ sơ đăng ký doanh nghiệp và cấp
    đăng ký doanh nghiệp; trường hợp hồ sơ chưa hợp lệ, Cơ quan đăng ký kinh doanh
    phải thông báo bằng văn bản nội dung cần sửa đổi, bổ sung cho người thành lập
    doanh nghiệp. Trường hợp từ chối đăng ký doanh nghiệp thì phải thông báo bằng
    văn bản cho người thành lập doanh nghiệp và nêu rõ lý do.

    6. Chính phủ quy định về hồ sơ, trình tự, thủ tục, liên thông trong đăng ký doanh
    nghiệp.

    Theo đó, hồ sơ đăng ký doanh nghiệp qua mạng thông tin điện tử có giá trị pháp
    lý tương đương hồ sơ đăng ký doanh nghiệp bằng bản giấy.

    '
  - 'Việc tài liệu trong hồ sơ đăng ký thành lập công ty cổ phần sản xuất ô tô chở
    khách có được làm bằng tiếng nước ngoài không, theo quy định tại Điều 10 Nghị
    định 01/2021/NĐ-CP như sau:

    Ngôn ngữ sử dụng trong hồ sơ đăng ký doanh nghiệp

    1. Các giấy tờ, tài liệu trong hồ sơ đăng ký doanh nghiệp được lập bằng tiếng
    Việt.

    2. Trường hợp hồ sơ đăng ký doanh nghiệp có tài liệu bằng tiếng nước ngoài thì
    hồ sơ phải có bản dịch tiếng Việt công chứng kèm theo tài liệu bằng tiếng nước
    ngoài.

    3. Trường hợp giấy tờ, tài liệu trong hồ sơ đăng ký doanh nghiệp được làm bằng
    tiếng Việt và tiếng nước ngoài thì bản tiếng Việt được sử dụng để thực hiện thủ
    tục đăng ký doanh nghiệp.

    Như vậy, tài liệu trong hồ sơ đăng ký thành lập công ty cổ phần sản xuất ô tô
    chở khách có thể làm bằng tiếng nước ngoài nhưng phải có bản dịch tiếng Việt công
    chứng kèm theo tài liệu bằng tiếng nước ngoài.

    '
  - 'Việc quyền sở hữu tài sản góp vốn được quy định tại Điều 35 Luật Doanh nghiệp
    2020 như sau:

    Chuyển quyền sở hữu tài sản góp vốn

    1. Thành viên công ty trách nhiệm hữu hạn, công ty hợp danh và cổ đông công ty
    cổ phần phải chuyển quyền sở hữu tài sản góp vốn cho công ty theo quy định sau
    đây:

    a) Đối với tài sản có đăng ký quyền sở hữu hoặc quyền sử dụng đất thì người góp
    vốn phải làm thủ tục chuyển quyền sở hữu tài sản đó hoặc quyền sử dụng đất cho
    công ty theo quy định của pháp luật. Việc chuyển quyền sở hữu, chuyển quyền sử
    dụng đất đối với tài sản góp vốn không phải chịu lệ phí trước bạ;

    b) Đối với tài sản không đăng ký quyền sở hữu, việc góp vốn phải được thực hiện
    bằng việc giao nhận tài sản góp vốn có xác nhận bằng biên bản, trừ trường hợp
    được thực hiện thông qua tài khoản.

    2. Biên bản giao nhận tài sản góp vốn phải bao gồm các nội dung chủ yếu sau đây:

    a) Tên, địa chỉ trụ sở chính của công ty;

    b) Họ, tên, địa chỉ liên lạc, số giấy tờ pháp lý của cá nhân, số giấy tờ pháp
    lý của tổ chức của người góp vốn;

    c) Loại tài sản và số đơn vị tài sản góp vốn; tổng giá trị tài sản góp vốn và
    tỷ lệ của tổng giá trị tài sản đó trong vốn điều lệ của công ty;

    d) Ngày giao nhận; chữ ký của người góp vốn hoặc người đại diện theo ủy quyền
    của người góp vốn và người đại diện theo pháp luật của công ty.

    ...

    Như vậy, đối với tài sản góp vốn là quyền sử dụng đất thì thành viên công ty trách
    nhiệm hữu hạn có trách nhiệm làm thủ tục chuyển quyền sử dụng đất của mình cho
    công ty theo quy định của pháp luật.

    Việc chuyển quyền sử dụng đất đối với tài sản góp vốn này sẽ không phải chịu lệ
    phí trước bạ.

    '
- source_sentence: Sau khi tiếp nhận hồ sơ chấm dứt hoạt động văn phòng đại diện thì
    Phòng Đăng ký kinh doanh gửi thông tin đến cơ quan nào?
  sentences:
  - 'Việc khởi kiện đối với Hội đồng quản trị được quy định tại khoản 1 Điều 166 Luật
    Doanh nghiệp 2020 như sau:

    Quyền khởi kiện đối với thành viên Hội đồng quản trị, Giám đốc, Tổng giám đốc

    1. Cổ đông, nhóm cổ đông sở hữu ít nhất 01% tổng số cổ phần phổ thông có quyền
    tự mình hoặc nhân danh công ty khởi kiện trách nhiệm cá nhân, trách nhiệm liên
    đới đối với các thành viên Hội đồng quản trị, Giám đốc hoặc Tổng giám đốc để yêu
    cầu hoàn trả lợi ích hoặc bồi thường thiệt hại cho công ty hoặc người khác trong
    trường hợp sau đây:

    a) Vi phạm trách nhiệm của người quản lý công ty theo quy định tại Điều 165 của
    Luật này;

    b) Không thực hiện, thực hiện không đầy đủ, thực hiện không kịp thời hoặc thực
    hiện trái với quy định của pháp luật hoặc Điều lệ công ty, nghị quyết, quyết định
    của Hội đồng quản trị đối với quyền và nghĩa vụ được giao;

    c) Lạm dụng địa vị, chức vụ và sử dụng thông tin, bí quyết, cơ hội kinh doanh,
    tài sản khác của công ty để tư lợi hoặc phục vụ lợi ích của tổ chức, cá nhân khác;

    d) Trường hợp khác theo quy định của pháp luật và Điều lệ công ty.

    ...

    Theo đó, nếu Hội đồng quản trị không triệu tập cuộc họp Đại hội đồng cổ đông bất
    thường trong trường hợp có yêu cầu triệu tập theo quy định dẫn đến thiệt hại cho
    công ty thì cổ đông sở hữu 2% cổ phần phổ thông có quyền tự mình hoặc nhân danh
    công ty khởi kiện trách nhiệm liên đới đối với các thành viên Hội đồng quản trị
    này.

    '
  - 'Việc hồ sơ đăng ký thành lập công ty TNHH 2 thành viên trở lên hoạt động dịch
    vụ trồng trọt có bắt buộc lập bằng tiếng Việt không, theo quy định tại Điều 10
    Nghị định 01/2021/NĐ-CP như sau:

    Ngôn ngữ sử dụng trong hồ sơ đăng ký doanh nghiệp

    1. Các giấy tờ, tài liệu trong hồ sơ đăng ký doanh nghiệp được lập bằng tiếng
    Việt.

    2. Trường hợp hồ sơ đăng ký doanh nghiệp có tài liệu bằng tiếng nước ngoài thì
    hồ sơ phải có bản dịch tiếng Việt công chứng kèm theo tài liệu bằng tiếng nước
    ngoài.

    3. Trường hợp giấy tờ, tài liệu trong hồ sơ đăng ký doanh nghiệp được làm bằng
    tiếng Việt và tiếng nước ngoài thì bản tiếng Việt được sử dụng để thực hiện thủ
    tục đăng ký doanh nghiệp.

    Như vậy, các giấy tờ, tài liệu trong hồ sơ đăng ký thành lập công ty TNHH 2 thành
    viên trở lên hoạt động dịch vụ trồng trọt được lập bằng tiếng Việt.

    Trường hợp hồ sơ có tài liệu bằng tiếng nước ngoài thì hồ sơ phải có bản dịch
    tiếng Việt công chứng kèm theo tài liệu bằng tiếng nước ngoài.

    '
  - 'Việc tiếp nhận hồ sơ chấm dứt hoạt động văn phòng đại diện được quy định tại
    khoản 3 Điều 72 Nghị định 01/2021/NĐ-CP như sau:

    Chấm dứt hoạt động chi nhánh, văn phòng đại diện, địa điểm kinh doanh

    ...

    3. Sau khi tiếp nhận hồ sơ chấm dứt hoạt động chi nhánh, văn phòng đại diện, địa
    điểm kinh doanh, Phòng Đăng ký kinh doanh gửi thông tin về việc chi nhánh, văn
    phòng đại diện, địa điểm kinh doanh chấm dứt hoạt động cho Cơ quan thuế. Trong
    thời hạn 02 ngày làm việc kể từ ngày nhận được thông tin của Phòng Đăng ký kinh
    doanh, Cơ quan thuế gửi ý kiến về việc hoàn thành nghĩa vụ nộp thuế của chi nhánh,
    văn phòng đại diện, địa điểm kinh doanh đến Phòng đăng ký kinh doanh. Trong thời
    hạn 05 ngày làm việc kể từ ngày nhận hồ sơ chấm dứt hoạt động chi nhánh, văn phòng
    đại diện, địa điểm kinh doanh, Phòng Đăng ký kinh doanh thực hiện chấm dứt hoạt
    động của chi nhánh, văn phòng đại diện, địa điểm kinh doanh trong Cơ sở dữ liệu
    quốc gia về đăng ký doanh nghiệp nếu không nhận được ý kiến từ chối của Cơ quan
    thuế, đồng thời ra thông báo về việc chấm dứt hoạt động chi nhánh, văn phòng đại
    diện, địa điểm kinh doanh.

    ...

    Như vậy, theo quy định, sau khi tiếp nhận hồ sơ chấm dứt hoạt động văn phòng đại
    diện thì Phòng Đăng ký kinh doanh phải gửi thông tin về việc văn phòng đại diện
    chấm dứt hoạt động cho Cơ quan thuế.

    Trong thời hạn 02 ngày làm việc kể từ ngày nhận được thông tin của Phòng Đăng
    ký kinh doanh, Cơ quan thuế gửi ý kiến về việc hoàn thành nghĩa vụ nộp thuế của
    văn phòng đại diện đến Phòng đăng ký kinh doanh.

    '
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- pearson_cosine
- spearman_cosine
model-index:
- name: SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2
  results:
  - task:
      type: semantic-similarity
      name: Semantic Similarity
    dataset:
      name: test evaluation
      type: test_evaluation
    metrics:
    - type: pearson_cosine
      value: .nan
      name: Pearson Cosine
    - type: spearman_cosine
      value: .nan
      name: Spearman Cosine
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Sau khi tiếp nhận hồ sơ chấm dứt hoạt động văn phòng đại diện thì Phòng Đăng ký kinh doanh gửi thông tin đến cơ quan nào?',
    'Việc tiếp nhận hồ sơ chấm dứt hoạt động văn phòng đại diện được quy định tại khoản 3 Điều 72 Nghị định 01/2021/NĐ-CP như sau:\nChấm dứt hoạt động chi nhánh, văn phòng đại diện, địa điểm kinh doanh\n...\n3. Sau khi tiếp nhận hồ sơ chấm dứt hoạt động chi nhánh, văn phòng đại diện, địa điểm kinh doanh, Phòng Đăng ký kinh doanh gửi thông tin về việc chi nhánh, văn phòng đại diện, địa điểm kinh doanh chấm dứt hoạt động cho Cơ quan thuế. Trong thời hạn 02 ngày làm việc kể từ ngày nhận được thông tin của Phòng Đăng ký kinh doanh, Cơ quan thuế gửi ý kiến về việc hoàn thành nghĩa vụ nộp thuế của chi nhánh, văn phòng đại diện, địa điểm kinh doanh đến Phòng đăng ký kinh doanh. Trong thời hạn 05 ngày làm việc kể từ ngày nhận hồ sơ chấm dứt hoạt động chi nhánh, văn phòng đại diện, địa điểm kinh doanh, Phòng Đăng ký kinh doanh thực hiện chấm dứt hoạt động của chi nhánh, văn phòng đại diện, địa điểm kinh doanh trong Cơ sở dữ liệu quốc gia về đăng ký doanh nghiệp nếu không nhận được ý kiến từ chối của Cơ quan thuế, đồng thời ra thông báo về việc chấm dứt hoạt động chi nhánh, văn phòng đại diện, địa điểm kinh doanh.\n...\nNhư vậy, theo quy định, sau khi tiếp nhận hồ sơ chấm dứt hoạt động văn phòng đại diện thì Phòng Đăng ký kinh doanh phải gửi thông tin về việc văn phòng đại diện chấm dứt hoạt động cho Cơ quan thuế.\nTrong thời hạn 02 ngày làm việc kể từ ngày nhận được thông tin của Phòng Đăng ký kinh doanh, Cơ quan thuế gửi ý kiến về việc hoàn thành nghĩa vụ nộp thuế của văn phòng đại diện đến Phòng đăng ký kinh doanh.\n',
    'Việc hồ sơ đăng ký thành lập công ty TNHH 2 thành viên trở lên hoạt động dịch vụ trồng trọt có bắt buộc lập bằng tiếng Việt không, theo quy định tại Điều 10 Nghị định 01/2021/NĐ-CP như sau:\nNgôn ngữ sử dụng trong hồ sơ đăng ký doanh nghiệp\n1. Các giấy tờ, tài liệu trong hồ sơ đăng ký doanh nghiệp được lập bằng tiếng Việt.\n2. Trường hợp hồ sơ đăng ký doanh nghiệp có tài liệu bằng tiếng nước ngoài thì hồ sơ phải có bản dịch tiếng Việt công chứng kèm theo tài liệu bằng tiếng nước ngoài.\n3. Trường hợp giấy tờ, tài liệu trong hồ sơ đăng ký doanh nghiệp được làm bằng tiếng Việt và tiếng nước ngoài thì bản tiếng Việt được sử dụng để thực hiện thủ tục đăng ký doanh nghiệp.\nNhư vậy, các giấy tờ, tài liệu trong hồ sơ đăng ký thành lập công ty TNHH 2 thành viên trở lên hoạt động dịch vụ trồng trọt được lập bằng tiếng Việt.\nTrường hợp hồ sơ có tài liệu bằng tiếng nước ngoài thì hồ sơ phải có bản dịch tiếng Việt công chứng kèm theo tài liệu bằng tiếng nước ngoài.\n',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.7755, 0.3505],
#         [0.7755, 1.0000, 0.3644],
#         [0.3505, 0.3644, 1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Semantic Similarity

* Dataset: `test_evaluation`
* Evaluated with [<code>EmbeddingSimilarityEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.EmbeddingSimilarityEvaluator)

| Metric              | Value   |
|:--------------------|:--------|
| pearson_cosine      | nan     |
| **spearman_cosine** | **nan** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 12,186 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                        | sentence_1                                                                           | label                                                          |
  |:--------|:----------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                            | string                                                                               | float                                                          |
  | details | <ul><li>min: 9 tokens</li><li>mean: 38.78 tokens</li><li>max: 81 tokens</li></ul> | <ul><li>min: 69 tokens</li><li>mean: 252.44 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.17</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                               | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | label            |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Công ty cổ phần chỉ còn lại một cổ đông thì phải tiến hành chuyển đổi thành công ty trách nhiệm hữu hạn một thành viên trong thời gian nào?</code> | <code>Căn cứ tại khoản 2 Điều 17 Luật Doanh nghiệp 2020 quyền thành lập, góp vốn, mua cổ phần, mua phần vốn góp và quản lý doanh nghiệp:<br>Quyền thành lập, góp vốn, mua cổ phần, mua phần vốn góp và quản lý doanh nghiệp<br>...<br>2. Tổ chức, cá nhân sau đây không có quyền thành lập và quản lý doanh nghiệp tại Việt Nam:<br>a) Cơ quan nhà nước, đơn vị lực lượng vũ trang nhân dân sử dụng tài sản nhà nước để thành lập doanh nghiệp kinh doanh thu lợi riêng cho cơ quan, đơn vị mình;<br>b) Cán bộ, công chức, viên chức theo quy định của Luật Cán bộ, công chức và Luật Viên chức;<br>c) Sĩ quan, hạ sĩ quan, quân nhân chuyên nghiệp, công nhân, viên chức quốc phòng trong các cơ quan, đơn vị thuộc Quân đội nhân dân Việt Nam; sĩ quan, hạ sĩ quan chuyên nghiệp, công nhân công an trong các cơ quan, đơn vị thuộc Công an nhân dân Việt Nam, trừ người được cử làm đại diện theo ủy quyền để quản lý phần vốn góp của Nhà nước tại doanh nghiệp hoặc quản lý tại doanh nghiệp nhà nước;<br>d) Cán bộ lãnh đạo, quản lý nghiệp vụ trong doanh ng...</code>    | <code>0.0</code> |
  | <code>Giám đốc công ty trách nhiệm hữu hạn hai thành viên là người miễn nhiệm kế toán trưởng đúng không?</code>                                          | <code>Trường hợp thu hồi Giấy chứng nhận đăng ký hoạt động chi nhánh được quy định tại khoản 1 Điều 77 Nghị định 01/2021/NĐ-CP như sau:<br>Thu hồi Giấy chứng nhận đăng ký hoạt động chi nhánh, văn phòng đại diện<br>1. Chi nhánh, văn phòng đại diện bị thu hồi Giấy chứng nhận đăng ký hoạt động chi nhánh, văn phòng đại diện trong các trường hợp sau đây:<br>a) Nội dung kê khai trong hồ sơ đăng ký hoạt động chi nhánh, văn phòng đại diện là giả mạo;<br>b) Chi nhánh, văn phòng đại diện ngừng hoạt động 01 năm mà không thông báo với Phòng Đăng ký kinh doanh và Cơ quan thuế;<br>c) Theo quyết định của Tòa án, đề nghị của cơ quan có thẩm quyền theo quy định của luật.<br>2. Trường hợp nội dung kê khai trong hồ sơ đăng ký thành lập mới chi nhánh, văn phòng đại diện là giả mạo thì Phòng Đăng ký kinh doanh ra thông báo về hành vi vi phạm của doanh nghiệp và ra quyết định thu hồi Giấy chứng nhận đăng ký hoạt động chi nhánh, văn phòng đại diện.<br>...<br>Theo đó, trường hợp chi nhánh ngừng hoạt động 01 năm mà không thông báo với Phòn...</code> | <code>0.0</code> |
  | <code>Trước khi mua trái phiếu doanh nghiệp, nhà đầu tư phải ký văn bản xác nhận những nội dung gì?</code>                                               | <code>Thời hiệu xử phạt hành vi không đăng tải thông tin về cơ sở dữ liệu về nhà thầu lên Báo đấu thầu được quy định tại khoản 1 Điều 5 Nghị định 122/2021/NĐ-CP như sau:<br>Thời hiệu và thời điểm xác định thời hiệu xử phạt vi phạm hành chính<br>1. Thời hiệu xử phạt vi phạm hành chính đối với lĩnh vực đầu tư, đấu thầu, đăng ký doanh nghiệp là 01 năm; đối với lĩnh vực quy hoạch là 02 năm.<br>2. Các hành vi vi phạm hành chính quy định tại Điều 7; Điều 9; Điều 10; Điều 13; Điều 14; khoản 2 Điều 15; khoản 3 Điều 16; Điều 17; Điều 18; Điều 19; Điều 20; Điều 21; Điều 22; Điều 23; Điều 24; Điều 30; Điều 36; Điều 37; Điều 43; Điều 44; Điều 45; Điều 46; Điều 47; Điều 48; Điều 49; Điều 50; Điều 51; Điều 52; Điều 53; Điều 54; Điều 55; Điều 56; Điều 57; Điều 58; Điều 59; Điều 60; Điều 61; Điều 62; Điều 63; Điều 64; Điều 65; Điều 66; Điều 67; Điều 68, Điều 69; Điều 70; Điều 71 và Điều 72 của Nghị định này là hành vi vi phạm hành chính đang thực hiện.<br>Đối với hành vi vi phạm đang thực hiện thì thời hiệu được tính t...</code>             | <code>0.0</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 128
- `per_device_eval_batch_size`: 128
- `num_train_epochs`: 30
- `fp16`: True
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 128
- `per_device_eval_batch_size`: 128
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 30
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: True
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch | Step | test_evaluation_spearman_cosine |
|:-----:|:----:|:-------------------------------:|
| 1.0   | 96   | nan                             |


### Framework Versions
- Python: 3.10.18
- Sentence Transformers: 5.1.2
- Transformers: 4.53.0
- PyTorch: 2.7.1+cu128
- Accelerate: 1.11.0
- Datasets: 4.4.1
- Tokenizers: 0.21.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->