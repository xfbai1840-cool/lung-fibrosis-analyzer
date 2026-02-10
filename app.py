import streamlit as st
import cv2
import numpy as np
import pandas as pd

# 设置网页标题
st.set_page_config(page_title="肺纤维化病理自动化分析平台", layout="wide")

def calculate_ashcroft(density):
    """根据组织密度估算 Ashcroft 评分"""
    if density <= 12: return 0.0
    score = (density * 2 - 12) * 8 / (75 - 12)
    return round(min(max(score, 0), 5), 0.1)

def process_image(uploaded_file):
    """处理上传的图片文件"""
    # 将上传的文件转为 OpenCV 格式
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # 图像处理逻辑
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 计算指标
    density = (np.count_nonzero(thresh) / thresh.size) * 100
    score = calculate_ashcroft(density)
    
    # 生成预览图 (左原图，右掩模)
    mask_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    preview_img = cv2.hconcat([img, mask_bgr])
    
    return density, score, preview_img

# --- 网页界面 ---
st.title("🔬 肺纤维化 (IPF) 病理切片自动分析系统")
st.markdown("上传小鼠肺部 H&E 染色切片，系统将自动计算组织密度并估算 Ashcroft 评分。")

uploaded_files = st.file_uploader("选择图片文件 (支持 JPG, PNG, TIF)", type=['jpg', 'jpeg', 'png', 'tif'], accept_multiple_files=True)

if uploaded_files:
    all_results = []
    
    for uploaded_file in uploaded_files:
        with st.expander(f"查看分析结果: {uploaded_file.name}", expanded=True):
            density, score, preview = process_image(uploaded_file)
            
            # 显示结果
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("组织密度", f"{density:.2f}%")
                st.metric("Ashcroft 评分", score)
            with col2:
                st.image(preview, caption=f"左：原图 | 右：识别区域", use_container_width=True)
            
            all_results.append({"文件名": uploaded_file.name, "密度(%)": density, "Ashcroft评分": score})
    
    # 汇总下载
    st.divider()
    df = pd.DataFrame(all_results)
    st.subheader("📊 汇总统计")
    st.dataframe(df)
    
    # 提供 CSV 下载按钮
    csv = df.to_csv(index=False).encode('utf_8_sig')

    st.download_button("下载分析报告 (CSV)", data=csv, file_name="pathology_report.csv", mime="text/csv")
