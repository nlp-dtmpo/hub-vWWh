import csv
import ast
from collections import Counter

# 定义标签映射
genre_mapping = {
    28: "Action",
    12: "Adventure",
    16: "Animation",
    35: "Comedy",
    80: "Crime",
    99: "Documentary",
    18: "Drama",
    10751: "Family",
    14: "Fantasy",
    36: "History",
    27: "Horror",
    10402: "Music",
    9648: "Mystery",
    10749: "Romance",
    878: "Science Fiction",
    10770: "TV Movie",
    53: "Thriller",
    10752: "War",
    37: "Western"
}

# 读取输入CSV文件
print("正在读取 movies_overview.csv...")
with open('movies_overview.csv', 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    movies = list(reader)

print(f"总共读取到 {len(movies)} 部电影")

# 第一步：为每部电影提取第一个标签，并创建临时列表
print("正在提取每部电影的第一个标签...")
temp_movies = []  # 存储(overview, 第一个标签)的临时列表
skipped_no_overview = 0
skipped_no_genres = 0

for i, movie in enumerate(movies, 1):
    if i % 500 == 0:
        print(f"提取进度: {i}/{len(movies)}...")

    # 检查是否有overview
    overview = movie.get('overview', '').strip()
    if not overview:
        skipped_no_overview += 1
        continue

    # 解析genre_ids字符串为列表
    try:
        genre_ids = ast.literal_eval(movie.get('genre_ids', '[]'))
    except (ValueError, SyntaxError):
        # 如果解析失败，尝试其他格式
        genre_ids_str = movie.get('genre_ids', '')
        if genre_ids_str.startswith('[') and genre_ids_str.endswith(']'):
            try:
                genre_ids = [int(id_str.strip()) for id_str in genre_ids_str[1:-1].split(',') if
                             id_str.strip().isdigit()]
            except:
                genre_ids = []
        else:
            genre_ids = []

    # 检查是否有标签
    if not genre_ids:
        skipped_no_genres += 1
        continue

    # 只取第一个标签
    first_genre_id = genre_ids[0]
    if first_genre_id in genre_mapping:
        temp_movies.append((overview, first_genre_id))

print(f"\n提取完成！")
print(f"有效电影（有overview和标签）: {len(temp_movies)} 部")
print(f"跳过（无overview）: {skipped_no_overview} 部")
print(f"跳过（无标签）: {skipped_no_genres} 部")

# 第二步：统计第一个标签的出现频率
print("\n正在统计第一个标签的出现频率...")
genre_counter = Counter()

for overview, genre_id in temp_movies:
    genre_counter[genre_id] += 1

print("第一个标签出现频率统计:")
# 筛选出现次数>5的标签
valid_genre_ids = {genre_id for genre_id, count in genre_counter.items() if count > 5}
invalid_genre_ids = {genre_id for genre_id, count in genre_counter.items() if count <= 5}

print(f"有效标签（出现次数>5）: {len(valid_genre_ids)} 个")
print(f"无效标签（出现次数≤5）: {len(invalid_genre_ids)} 个")

# 显示被过滤掉的标签
if invalid_genre_ids:
    print("\n被过滤掉的标签（出现次数≤5）:")
    for genre_id in sorted(invalid_genre_ids):
        if genre_id in genre_mapping:
            print(f"  {genre_mapping[genre_id]} (ID: {genre_id}): 出现 {genre_counter[genre_id]} 次")

# 显示保留的标签
print("\n保留的标签（出现次数>5）:")
for genre_id in sorted(valid_genre_ids):
    if genre_id in genre_mapping:
        print(f"  {genre_mapping[genre_id]} (ID: {genre_id}): 出现 {genre_counter[genre_id]} 次")

# 第三步：根据标签频率过滤电影并写入文件
print("\n正在根据标签频率过滤并写入 movie_dataset.csv...")
with open('movie_dataset.csv', 'w', newline='', encoding='utf-8') as csvfile:
    # 创建写入器，使用制表符分隔
    writer = csv.writer(csvfile, delimiter='\t')

    processed_count = 0
    skipped_invalid_genre = 0

    # 处理临时电影列表
    for i, (overview, genre_id) in enumerate(temp_movies, 1):

        # 检查标签是否有效（出现次数>5）
        if genre_id not in valid_genre_ids:
            skipped_invalid_genre += 1
            continue

        # 获取标签名称
        genre_name = genre_mapping[genre_id]

        # 写入行（只有一个标签）
        writer.writerow([overview, genre_name])
        processed_count += 1

print("\n处理完成！")
print("=" * 50)
print(f"初始电影总数: {len(movies)} 部")
print(f"第一步提取后: {len(temp_movies)} 部（有overview和标签）")
print(f"第二步过滤后: {processed_count} 部（标签出现次数>5）")
print("-" * 30)
print(f"跳过统计:")
print(f"  无overview: {skipped_no_overview} 部")
print(f"  无标签: {skipped_no_genres} 部")
print(f"  标签出现次数≤5: {skipped_invalid_genre} 部")
print("=" * 50)
print(f"文件 'movie_dataset.csv' 已成功创建！")

# 显示最终标签分布
print("\n最终数据集标签分布（按出现次数降序）:")
final_genre_counter = Counter()
for genre_id in valid_genre_ids:
    if genre_id in genre_mapping:
        final_genre_counter[genre_mapping[genre_id]] = 0

# 统计最终数据集中的标签分布
with open('movie_dataset.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)  # 跳过标题行
    for row in reader:
        if len(row) >= 2:
            genre_name = row[1]
            final_genre_counter[genre_name] += 1

# 按出现次数降序排序显示
print(f"总共 {len(final_genre_counter)} 个标签:")
for genre_name, count in sorted(final_genre_counter.items(), key=lambda x: x[1], reverse=True):
    print(f"  {genre_name}: {count} 部电影 ({count / processed_count * 100:.1f}%)")

# 显示标签数量统计
print(f"\n标签数量统计:")
print(f"  最多的标签: {max(final_genre_counter.values())} 部电影")
print(f"  最少的标签: {min(final_genre_counter.values())} 部电影")
print(f"  平均每标签: {processed_count / len(final_genre_counter):.1f} 部电影")


# import csv
# import sys
#
#
# def validate_csv_file(filename):
#     """
#     验证CSV文件是否符合要求
#     """
#     issues = []
#
#     try:
#         with open(filename, 'r', encoding='utf-8') as file:
#             # 使用制表符作为分隔符
#             reader = csv.reader(file, delimiter='\t')
#
#             for line_num, row in enumerate(reader, 1):
#                 # 检查每行的列数
#                 if len(row) != 2:
#                     issues.append(f"第 {line_num} 行: 列数不正确。期望2列，实际{len(row)}列")
#                     continue
#
#                 # 检查第一列（电影描述）是否为空
#                 if not row[0].strip():
#                     issues.append(f"第 {line_num} 行: 电影描述为空")
#
#                 # 检查第二列（类型）是否为空
#                 if not row[1].strip():
#                     issues.append(f"第 {line_num} 行: 电影类型为空")
#
#                 # 检查第二列是否符合格式（以逗号分隔的类型）
#                 genres = row[1].strip()
#                 if ',' in genres:
#                     # 检查每个类型是否都是有效的（至少包含非空格字符）
#                     genre_list = [g.strip() for g in genres.split(',')]
#                     for i, genre in enumerate(genre_list):
#                         if not genre:
#                             issues.append(f"第 {line_num} 行: 第{i + 1}个类型为空")
#                 else:
#                     # 只有一个类型的情况
#                     if not genres:
#                         issues.append(f"第 {line_num} 行: 电影类型格式不正确")
#
#         return issues
#
#     except FileNotFoundError:
#         return [f"错误: 文件 '{filename}' 未找到"]
#     except Exception as e:
#         return [f"错误: 读取文件时发生异常 - {str(e)}"]
#
#
# def main():
#     filename = "./movie_dataset.csv"
#     if filename:
#         issues = validate_csv_file(filename)
#
#         if not issues:
#             print("✓ 文件中的所有行都符合要求！")
#         else:
#             print("✗ 发现问题：")
#             for issue in issues:
#                 print(f"  - {issue}")
#
#
# if __name__ == "__main__":
#     main()