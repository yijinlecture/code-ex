import os
import re

"""
12.1.3.142025723451.jpg -> 12.1.3.14_20250702030405_1.jpg
{ip주소}_{타임스탬프}_{번호}.jpg 형태로 변환
"""


def parse_variable_datetime(s):
    """
    가변 길이의 날짜/시간 문자열('2591011')을 파싱하는 함수
    """
    parts = []
    max_vals = [31, 23, 59, 59] # 일, 시, 분, 초의 최댓값

    for max_val in max_vals:
        if not s: break
        
        if len(s) >= 2 and 0 <= int(s[:2]) <= max_val:
            parts.append(s[:2])
            s = s[2:]
        else:
            parts.append(s[:1])
            s = s[1:]
    
    while len(parts) < 4: parts.append('0')
        
    return parts[0], parts[1], parts[2], parts[3] # day, hour, minute, second

def convert_filename_final(filename):
    """
    서로 다른 두 가지 파일 형식을 모두 정확하게 변환하는 최종 함수
    """
    try:
        name, ext = os.path.splitext(filename)
        match = re.search(r"20\d{2}", name)
        
        if not match: return None
            
        split_index = match.start()
        prefix = name[:split_index]
        timestamp_part = name[split_index:]

        if not timestamp_part.isdigit(): return None

        year = timestamp_part[:4]
        
        if timestamp_part[4:6] in ['10', '11', '12']:
            month = timestamp_part[4:6]
            rest = timestamp_part[6:]
        else:
            month = timestamp_part[4:5]
            rest = timestamp_part[5:]

        
        if len(rest) == 5: 
            day = rest[0]
            hour = rest[1]
            minute = rest[2]
            second = rest[3]
            index = rest[4]
        else:
            datetime_s = rest[:-1]
            index = rest[-1]
            day, hour, minute, second = parse_variable_datetime(datetime_s)
            

        
        new_timestamp = (
            f"{year}"
            f"{int(month):02d}"
            f"{int(day):02d}"
            f"{int(hour):02d}"
            f"{int(minute):02d}"
            f"{int(second):02d}"
            f"_{index}"  # <----타임스탬프_인덱스 (형식 수정 가능!!!)
        )
        
        if prefix:
            new_name = f"{prefix}_{new_timestamp}" # <----ip주소_타임스탬프 (형식 수정 가능!!!)
        else:
            new_name = new_timestamp
            
        return new_name + ext

    except (IndexError, ValueError):
        return None

def bulk_rename_files(folder_path):
    """
    지정된 폴더 경로에 있는 모든 파일의 이름을 규칙에 따라 변경합니다.
    """
    if not os.path.isdir(folder_path):
        print(f"오류: '{folder_path}'는 유효한 폴더가 아닙니다. 경로를 확인해주세요.")
        return

    files = os.listdir(folder_path)
    renamed_count = 0
    skipped_count = 0

    print(f"'{folder_path}' 폴더의 파일명 변환 시작...")
    print("-" * 30)

    for filename in files:
        old_path = os.path.join(folder_path, filename)

        # 폴더가 아닌 '파일'인 경우에만 처리
        if os.path.isfile(old_path):
            new_filename = convert_filename_final(filename)
            
            if new_filename and new_filename != filename:
                new_path = os.path.join(folder_path, new_filename)
                
                if os.path.exists(new_path):
                    print(f"경고: '{new_filename}' 파일이 이미 존재하여 건너뜁니다.")
                    skipped_count += 1
                    continue

                os.rename(old_path, new_path)
                print(f"'{filename}'  ->  '{new_filename}'")
                renamed_count += 1
            else:
                skipped_count += 1
    
    print("-" * 30)
    print("\n--- 작업 완료 ---")
    print(f"총 {renamed_count}개의 파일명을 변경했습니다.")
    print(f"총 {skipped_count}개의 파일을 건너뛰었습니다.")

if __name__ == "__main__":
    
    TARGET_FOLDER_PATH = "파일이 들어있는 폴더 경로"  # <--- 이 부분을 수정하세요!
    
    bulk_rename_files(TARGET_FOLDER_PATH)
