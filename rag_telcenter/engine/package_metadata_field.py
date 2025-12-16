class PackageMetadataField:
    def __init__(self, field_name: str, field_description: str, field_type: str):
        self.field_name = field_name
        self.field_description = field_description
        self.field_type = field_type
        if field_type not in ["number", "text"]:
            raise ValueError(f"Invalid field_type: {field_type}, must be 'number' or 'text'")


"""
- Mã dịch vụ: Mã định danh duy nhất của gói cước, ví dụ "SD70".
- Thời gian thanh toán: Thường có hai giá trị "Trả trước" hoặc "Trả sau".
- Các dịch vụ tiên quyết: Các dịch vụ cần có trước khi đăng ký gói cước này, có thể để trống.
- Giá (VNĐ): Giá của gói cước trong một chu kỳ, tính theo đồng Việt Nam.
- Chu kỳ (ngày): Thời gian hiệu lực của gói cước tính theo ngày. Hết chu kỳ sẽ phải gia hạn để tiếp tục sử dụng.
- 4G tốc độ tiêu chuẩn/ngày: Dung lượng dữ liệu 4G tốc độ tiêu chuẩn mà người dùng nhận được mỗi ngày, được biểu hiện bằng số GB. Nếu sử dụng hết sẽ bị giảm tốc độ.
- 4G tốc độ cao/ngày
- 4G tốc độ tiêu chuẩn/chu kỳ
- 4G tốc độ cao/chu kỳ
- Gọi nội mạng: Chi tiết ưu đãi gọi nội mạng trong chu kỳ, ví dụ "Miễn phí 30 phút gọi"
- Gọi ngoại mạng
- Tin nhắn: Chi tiết ưu đãi tin nhắn trong chu kỳ.
- Chi tiết: Mô tả thêm về gói cước, bao gồm các ưu đãi, điều kiện sử dụng, giới hạn...
- Tự động gia hạn: Cho biết gói cước có tự động gia hạn sau khi hết chu kỳ hay không. Nhận giá trị "Có" hoặc "Không".
- Cú pháp đăng ký: Hướng dẫn cú pháp SMS hoặc thao tác để đăng ký gói cước.
"""
PACKAGE_FIELDS = [
    PackageMetadataField(
        field_name="Nhà mạng",
        field_description="Tên nhà mạng cung cấp gói cước, ví dụ 'Viettel', 'Mobifone', 'Vinaphone'.",
        field_type="text",
    ),
    
    PackageMetadataField(
        field_name="Mã dịch vụ",
        field_description=" Mã định danh duy nhất của gói cước, ví dụ 'SD70'.",
        field_type="text",
    ),

    PackageMetadataField(
        field_name="Thời gian thanh toán",
        field_description="Hình thức thanh toán của gói: 'Trả trước' hoặc 'Trả sau'.",
        field_type="text",
    ),
    PackageMetadataField(
        field_name="Các dịch vụ tiên quyết",
        field_description="Các dịch vụ cần có trước khi đăng ký gói (có thể để trống).",
        field_type="text",
    ),
    PackageMetadataField(
        field_name="Giá (VNĐ)",
        field_description="Giá của gói cước trong một chu kỳ, tính theo đồng Việt Nam.",
        field_type="number",
    ),
    PackageMetadataField(
        field_name="Chu kỳ (ngày)",
        field_description="Thời gian hiệu lực của gói cước tính theo ngày.",
        field_type="number",
    ),
    PackageMetadataField(
        field_name="4G tốc độ tiêu chuẩn/ngày",
        field_description="Dung lượng data 4G tốc độ tiêu chuẩn nhận được mỗi ngày (ví dụ '1GB').",
        field_type="text",
    ),
    PackageMetadataField(
        field_name="4G tốc độ cao/ngày",
        field_description="Dung lượng 4G tốc độ cao mỗi ngày (ví dụ '3GB').",
        field_type="text",
    ),
    PackageMetadataField(
        field_name="4G tốc độ tiêu chuẩn/chu kỳ",
        field_description="Dung lượng 4G tốc độ tiêu chuẩn cho cả chu kỳ (ví dụ '30GB').",
        field_type="text",
    ),
    PackageMetadataField(
        field_name="4G tốc độ cao/chu kỳ",
        field_description="Dung lượng 4G tốc độ cao cho cả chu kỳ.",
        field_type="text",
    ),
    PackageMetadataField(
        field_name="Gọi nội mạng",
        field_description="Chi tiết ưu đãi gọi nội mạng trong chu kỳ (ví dụ 'Miễn phí 30 phút').",
        field_type="text",
    ),
    PackageMetadataField(
        field_name="Gọi ngoại mạng",
        field_description="Chi tiết ưu đãi gọi ngoại mạng trong chu kỳ.",
        field_type="text",
    ),
    PackageMetadataField(
        field_name="Tin nhắn",
        field_description="Chi tiết ưu đãi tin nhắn trong chu kỳ.",
        field_type="text",
    ),
    PackageMetadataField(
        field_name="Chi tiết",
        field_description="Mô tả thêm về gói cước: ưu đãi, điều kiện sử dụng, giới hạn, ghi chú...",
        field_type="text",
    ),
    PackageMetadataField(
        field_name="Tự động gia hạn",
        field_description="Cho biết gói có tự động gia hạn sau khi hết chu kỳ hay không ('Có'/'Không').",
        field_type="text",
    ),
    PackageMetadataField(
        field_name="Cú pháp đăng ký",
        field_description="Hướng dẫn cú pháp SMS hoặc thao tác để đăng ký gói cước.",
        field_type="text",
    ),
]

import pyarrow as pa

PACKAGE_SCHEMA = pa.schema([
    pa.field(field.field_name, pa.float32() if field.field_type == "number" else pa.string())
    for field in PACKAGE_FIELDS
])
