import streamlit as st
import cv2
import numpy as np
import pandas as pd

# 设置网页标题
st.set_page_config(page_title="肺纤维化病理自动化分析平台", layout="wide")

def calculate_custom_score(density):
    """
    根据用户定义的区间进行评分:
    组织密度 >= 36% -> 分值为 5.0
    组织密度 <= 34% -> 分值为 0.1
    34% < 密度 < 36% -> 线性插值
    """
    if density >= 36:
        return 5.0
    if density <= 34:
        return 0.1
    
    # 线性插值逻辑: 在 2% 的密度区间内映射 4.9 分的分差
    score = (density - 34) * (5.0 - 0.1) / (36 - 34) + 0.1
    return round(score, 2)

def process_image(uploaded_file):
    """处理上传的图片文件"""
    # 将上传的文件转为 OpenCV 格式
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        return None, None, None
    
    # 图像处理逻辑：转换为灰度并使用大津法二值化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 计算指标
    tissue_pixels = np.count_nonzero(thresh)
    total_pixels = thresh.size
    density = (tissue_pixels / total_pixels) * 100
    score = calculate_custom_score(density)
    
    # 生成预览图 (左原图，右掩模)
    mask_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    preview_img = cv2.hconcat([img, mask_bgr])
    
    return density, score, preview_img

# --- 网页界面 ---
st.title("🔬 肺纤维化 (IPF) 病理切片自动分析系统")
st.markdown("""
通过上传小鼠肺部 H&E 染色切片，系统将自动识别组织区域并计算评分。
**当前评分标准：**
- 组织密度 **≥ 36%**：分值为 **5.0**
- 组织密度 **≤ 34%**：分值为 **0.1**
""")

uploaded_files = st.file_uploader("选择图片文件 (支持 JPG, PNG, TIF)", type=['jpg', 'jpeg', 'png', 'tif'], accept_multiple_files=True)

if uploaded_files:
    all_results = []
    
    for uploaded_file in uploaded_files:
        # 使用 expander 展示每张图的详细结果
        with st.expander(f"查看分析结果: {uploaded_file.name}", expanded=True):
            density, score, preview = process_image(uploaded_file)
            
            if density is not None:
                # 布局显示指标和图片
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("组织密度", f"{density:.2f}%")
                    st.metric("校准分值 (0.1-5.0)", score)
                with col2:
                    st.image(preview, caption=f"分析对比 (左：原图 | 右：识别出的组织区域)", use_container_width=True)
                
                all_results.append({
                    "文件名": uploaded_file.name, 
                    "组织密度(%)": round(density, 2), 
                    "评分": score
                })
    
    # 汇总数据展示与下载
    if all_results:
        st.divider()
        df = pd.DataFrame(all_results)
        st.subheader("📊 汇总统计表")
        st.dataframe(df, use_container_width=True)
        
        # 提供 CSV 下载
        csv = df.to_csv(index=False).encode('utf_8_sig')
        st.download_button(
            label="下载完整分析报告 (CSV)",
            data=csv,
            file_name="lung_fibrosis_analysis.csv",
            mime="text/csv"
        )
