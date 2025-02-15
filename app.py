import time
import streamlit as st
import pandas as pd
import base64
from PIL import Image
import io
import re
from collections import Counter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

st.set_page_config(
    page_title="AccuNutri",
    page_icon="🔥",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255,75,75,0.4); }
        70% { box-shadow: 0 0 0 15px rgba(255,75,75,0); }
        100% { box-shadow: 0 0 0 0 rgba(255,75,75,0); }
    }

    .uploaded-image {
        border-radius: 20px;
        padding: 15px;
        background: linear-gradient(45deg, #ff4b4b, #ff6b6b);
        animation: pulse 2s infinite;
        margin: 20px 0;
    }

    .metric-card {
        padding: 1.5rem;
        border-radius: 15px;
        background: linear-gradient(145deg, #2b2b2b, #1a1a1a);
        color: white;
        margin: 1rem 0;
        border: 1px solid #ff4b4b;
        box-shadow: 0 4px 20px rgba(255,75,75,0.2);
    }

    .metric-value {
        font-size: 28px;
        font-weight: 800;
        background: linear-gradient(45deg, #ff4b4b, #ff9f43);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }

    .metric-label {
        color: #ffffffaa;
        font-size: 16px;
        letter-spacing: 0.5px;
    }

    .health-meter {
        width: 100%;
        height: 15px;
        background: #2b2b2b;
        border-radius: 10px;
        margin: 2rem 0;
        overflow: hidden;
    }

    .health-progress {
        height: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #ff4b4b, #ff9f43);
        transition: width 0.5s ease;
    }

    .stSpinner > div {
        border-color: #ff4b4b transparent transparent transparent !important;
    }
    </style>
""", unsafe_allow_html=True)


class NutritionExpert:
    def __init__(self, food_data_path):
        self.food_data = pd.read_csv(food_data_path)
        self.text_llm = OllamaLLM(model="llama3.2:latest")
        self.vision_llm = OllamaLLM(model="llama3.2-vision:latest")
        self.embeddings = HuggingFaceEmbeddings()
        self.analysis_runs = []

        self.vectorstore = FAISS.from_texts(
            [self._create_nutri_document(row) for _, row in self.food_data.iterrows()],
            self.embeddings
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.text_llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 10}),
            chain_type_kwargs={"prompt": PromptTemplate.from_template("""
                [INST] You are a clinical nutritionist. Analyze this meal: {question}
                Context database: {context}
                Required output format:
                CALORIES: [number] kcal
                CARBS: [number]g
                PROTEIN: [number]g
                FAT: [number]g
                HEALTH_SCORE: [1-10] (10=healthiest)
                REASON: [2-3 key factors]
                Ensure the output is strictly in this format. Do not include any additional text or explanations. [/INST]""")}
        )

    def _create_nutri_document(self, row):
        return (f"Food: {row['meal_description']}\n"
                f"Nutrition: {row['carb']}g carbs, {row['protein']}g protein, "
                f"{row['fat']}g fat, {row['energy']} kcal\n"
                f"Category: {'healthy' if row['energy'] < 400 else 'indulgent'}")

    def _analyze_image(self, image_file):
        with Image.open(image_file) as img:
            if img.mode == 'RGBA':
                img = img.convert('RGB')

            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

        response = self.vision_llm.generate(
            prompts=["""[INST] <image>
            Describe ONLY the food in this image. Include:
            - The type of food
            - The main ingredients
            - The portion size (if visible)
            - Any visible preparation or cooking method
            Do not comment on the photo quality, background, or anything unrelated to the food itself. [/INST]"""],
            images=[img_str]
        )
        return response.generations[0][0].text


@st.cache_resource
def load_analyzer():
    return NutritionExpert("meal_database.csv")


def parse_analysis_result(result_text):
    results = {}
    for line in result_text.split('\n'):
        line = line.strip()
        if not line:
            continue
        if ':' in line:
            key, val = line.split(':', 1)
            key = key.strip().replace(" ", "_")
            val = val.strip()
            results[key] = val
    return results


def clean_numerical_value(value):
    if isinstance(value, str):
        value = re.sub(r"[^0-9.]", "", value)
    return value


def validate_results(results):
    required_keys = ["CALORIES", "CARBS", "PROTEIN", "FAT", "HEALTH_SCORE", "REASON"]
    for key in required_keys:
        if key not in results:
            st.warning(f"Missing key in results: {key}")
            return False

    try:
        results["CALORIES"] = clean_numerical_value(results["CALORIES"])
        results["CARBS"] = clean_numerical_value(results["CARBS"])
        results["PROTEIN"] = clean_numerical_value(results["PROTEIN"])
        results["FAT"] = clean_numerical_value(results["FAT"])

        float(results["CALORIES"])
        float(results["CARBS"])
        float(results["PROTEIN"])
        float(results["FAT"])

        health_score = extract_numerical_value(results["HEALTH_SCORE"])
        if health_score is None or not (1 <= health_score <= 10):
            st.warning(f"Invalid HEALTH_SCORE: {results['HEALTH_SCORE']}")
            return False
    except (ValueError, AttributeError) as e:
        st.warning(f"Validation error: {e}")
        return False
    return True


def extract_numerical_value(value):
    if isinstance(value, str):
        match = re.search(r'\d+', value)
        if match:
            return float(match.group())
    return None


def run_analysis_multiple_times(analyzer, image_file, num_runs=5):
    results_list = []
    analyzer.analysis_runs = []

    for i in range(num_runs):
        try:
            vision_output = analyzer._analyze_image(image_file)
            analysis = analyzer.qa_chain.invoke(vision_output)
            raw_output = analysis['result']
            parsed_results = parse_analysis_result(raw_output)
            parsed_results["VISION_DESCRIPTION"] = vision_output

            analyzer.analysis_runs.append({
                "raw_output": raw_output,
                "parsed_results": parsed_results
            })

            if validate_results(parsed_results):
                results_list.append(parsed_results)
        except Exception as e:
            st.error(f"Error during analysis (Run {i + 1}): {e}")
            continue

    if not results_list:
        st.error("All analysis attempts failed. Raw outputs were not valid.")
        return None

    averaged_results = {}
    for key in results_list[0].keys():
        if key == "REASON":
            averaged_results[key] = results_list[0][key]
        else:
            values = [r[key] for r in results_list if key in r]
            if values:
                counter = Counter(values)
                most_common_value = counter.most_common(1)[0][0]
                averaged_results[key] = most_common_value

    return averaged_results


def calculate_health_score(calories, carbs, protein, fat):
    calorie_score = max(0, 10 - (calories / 200))
    carb_score = max(0, 10 - (carbs / 20))
    protein_score = (protein / 10)
    fat_score = max(0, 10 - (fat / 10))

    health_score = (calorie_score * 0.4) + (carb_score * 0.2) + (protein_score * 0.3) + (fat_score * 0.1)
    return max(1, min(round(health_score), 10))


def run_analysis_with_progress(analyzer, image_file, num_runs=5):
    results_list = []
    analyzer.analysis_runs = []

    progress_bar = st.progress(0)
    status_text = st.empty()
    cot_container = st.container()
    start_time = time.time()

    for i in range(num_runs):
        try:
            elapsed_time = time.time() - start_time
            avg_time_per_run = elapsed_time / (i + 1e-9)
            eta = avg_time_per_run * (num_runs - i - 1)

            progress = (i + 1) / num_runs
            progress_bar.progress(progress)
            status_text.markdown(f"""
                **Run {i + 1}/{num_runs}**  
                Elapsed: {elapsed_time:.1f}s  
                ETA: {eta:.1f}s remaining
            """)

            with cot_container:
                st.subheader("Real-Time Analysis Thinking")
                thinking_placeholder = st.empty()

                thinking_placeholder.markdown("🔍 Analyzing image...")
                vision_output = analyzer._analyze_image(image_file)
                thinking_placeholder.markdown(f"**Vision Model Description:**\n{vision_output}")

                thinking_placeholder.markdown("🧠 Processing nutritional analysis...")
                analysis = analyzer.qa_chain.invoke(vision_output)
                raw_output = analysis['result']
                thinking_placeholder.markdown(f"**Raw Analysis Output:**\n```\n{raw_output}\n```")

                parsed_results = parse_analysis_result(raw_output)
                parsed_results["VISION_DESCRIPTION"] = vision_output
                analyzer.analysis_runs.append({
                    "raw_output": raw_output,
                    "parsed_results": parsed_results
                })

                if validate_results(parsed_results):
                    results_list.append(parsed_results)
                    thinking_placeholder.markdown("✅ Analysis validated!")
                else:
                    thinking_placeholder.markdown("❌ Validation failed, retrying...")

                st.write("---")

        except Exception as e:
            st.error(f"Error during analysis (Run {i + 1}): {e}")
            continue

    progress_bar.empty()
    status_text.empty()

    if not results_list:
        st.error("All analysis attempts failed. Raw outputs were not valid.")
        return None
    averaged_results = {}
    for key in results_list[0].keys():
        if key == "REASON":
            averaged_results[key] = results_list[0][key]
        else:
            values = [r[key] for r in results_list if key in r]
            if values:
                counter = Counter(values)
                most_common_value = counter.most_common(1)[0][0]
                averaged_results[key] = most_common_value

    return averaged_results

def main():
    st.title("🔥 AccuNutri AI")
    st.subheader("Snap Your Meal for Instant Nutrition Analysis")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is not None:
        start_time = time.time()

        with st.container():
            st.markdown(
                f'<div class="uploaded-image"><img src="data:image/jpeg;base64,{base64.b64encode(uploaded_file.getvalue()).decode()}" style="width:100%; border-radius:10px;"></div>',
                unsafe_allow_html=True
            )

        analyzer = load_analyzer()

        with st.spinner("🔍 Analyzing your meal..."):
            results = run_analysis_with_progress(analyzer, uploaded_file, num_runs=5)

        if results is None:
            st.error("Unable to analyze the meal. Please try again or upload a different image.")
            return

        for key in ["CALORIES", "CARBS", "PROTEIN", "FAT"]:
            if key in results:
                results[key] = str(round(float(clean_numerical_value(results[key]))))

        calories = float(results.get("CALORIES", 0))
        carbs = float(results.get("CARBS", 0))
        protein = float(results.get("PROTEIN", 0))
        fat = float(results.get("FAT", 0))
        health_score = calculate_health_score(calories, carbs, protein, fat)
        results["HEALTH_SCORE"] = str(health_score)

        st.subheader("📊 Nutrition Analysis Results")

        st.markdown(
            f'<div class="health-meter"><div class="health-progress" style="width: {health_score * 10}%"></div></div>',
            unsafe_allow_html=True
        )
        st.subheader(f"Health Score: {health_score}/10")

        col1, col2 = st.columns(2)
        metrics = {
            "CALORIES": "Calories",
            "CARBS": "Carbs",
            "PROTEIN": "Protein",
            "FAT": "Fats"
        }

        for key, label in metrics.items():
            value = results.get(key, '--')
            if key in ["CARBS", "PROTEIN", "FAT"] and 'g' not in value:
                value += 'g'
            (col1 if key in ["CALORIES", "PROTEIN"] else col2).markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>""",
                unsafe_allow_html=True
            )

        if 'REASON' in results:
            with st.expander("💡 Key Factors"):
                st.write(results['REASON'])

        with st.expander("🧠 Full Chain of Thought"):
            st.write("### Vision Model Description")
            st.write(results.get("VISION_DESCRIPTION", "No description available."))

            st.write("### Analysis Process")
            for i, run in enumerate(analyzer.analysis_runs, start=1):
                st.write(f"#### Run {i} Raw Output")
                st.write(run["raw_output"])
                st.write(f"#### Run {i} Parsed Results")
                st.write(run["parsed_results"])

            st.write("### Final Analysis Results")
            st.write(results)

        thinking_time = time.time() - start_time
        st.write(f"⏱️ Total thinking time: {round(thinking_time, 2)} seconds")

    else:
        st.info("Please upload an image of your meal to get started.")

if __name__ == "__main__":
    main()