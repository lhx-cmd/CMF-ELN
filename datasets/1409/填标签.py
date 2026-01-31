import os

def generate_entity_file():
    filename = 'entity.txt'
    
    # 定义配置： (标签内容, 重复次数)
    # 数据来源：
    # 0.1409 -> 标签 0, 1409行
    # 1.1333 -> 标签 1, 1333行
    # 2.1024 -> 标签 2, 1024行
    # 3.158  -> 标签 3, 158行
    # 4.397  -> 标签 4, 397行
    data_config = [
        (0, 1710),
        (1, 1415),
        (2, 1024),
        (3, 163),
        (4, 462)
    ]

    print(f"正在生成 {filename} ...")
    
    total_lines = 0
    
    try:
        with open(filename, 'w') as f:
            for label, count in data_config:
                # 循环写入指定次数
                for _ in range(count):
                    f.write(f"{label}\n")
                
                print(f"已写入标签 {label}: {count} 行")
                total_lines += count
                
        print("-" * 30)
        print(f"成功！文件已保存至: {os.path.abspath(filename)}")
        print(f"共写入总行数: {total_lines}")
        
    except IOError as e:
        print(f"写入文件时发生错误: {e}")

if __name__ == "__main__":
    generate_entity_file()