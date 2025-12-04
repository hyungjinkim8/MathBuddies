import streamlit as st
import pandas as pd
import numpy as np
import sys, os
import json
import subprocess, glob, time
import random
from datetime import datetime
import altair as alt

# ============================================================================
#  GLOBAL STREAMLIT THEME
# ============================================================================

st.set_page_config(
    page_title="Welcome to MathBuddies!",
    page_icon="🧮",
    layout="centered"
)

st.markdown("""
<style>
    body { background-color: #FFF9EB; }
    .stProgress > div > div > div {
        background-color: #FF7B00 !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
#  HEADER SECTION
# ============================================================================

def app_header():
    st.markdown("""
        <div style="text-align: center; margin-top:-40px;">
            <h1 style="font-size: 3rem; color:#FF7B00; margin-bottom: 0;">
                🧮 MathBuddies!
            </h1>
            <p style="font-size: 1.2rem; color:#444;">
                Smart, Fun Math Practice for Awesome 3rd Graders!
            </p>
        </div>
    """, unsafe_allow_html=True)


# ============================================================================
#  IMPORT CLASSES
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SESSION_DATA_DIR = os.path.join(BASE_DIR, "session_data")

from mwp_classes import PromptGenerator, ProblemEvaluator, MWPGenerator, MWPSession

st.set_page_config(page_title="Adaptive Math Practice", page_icon="🧮")

# ============================================================================
# LOGIN SYSTEM
# ============================================================================

USER_CREDENTIALS = {
    "apple": "apple1",
    "peach": "peach3",
    "orange": "orange2"
}

def ensure_login():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = None

    if st.session_state.logged_in:
        with st.sidebar:
            st.success(f"✅ Logged in as **{st.session_state.username}**")
            if st.button("Logout"):
                st.session_state.logged_in = False
                st.session_state.username = None
                st.rerun()
        return True

    # -------------------------------
    # Login page with introduction
    # -------------------------------
    app_header()

    st.markdown("""
    <div style="
        background-color: #FFFFFF; padding: 20px;
        border-radius: 16px; border: 2px solid #FFE1B3;
        box-shadow: 0 4px 12px rgba(255,123,0,0.1);
        margin: 20px auto; max-width: 580px;
    ">
        <h3 style="color:#FF7B00; text-align:center;">✨ What’s Inside?</h3>
        <ul style="font-size: 1.1rem; line-height:1.6;">
            <li>🔢 Fresh, fun word problems every time</li>
            <li>🧠 Difficulty that adapts to your level</li>
            <li>⚡ Instant feedback with explanations</li>
            <li>📈 Track your progress and level up!</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Login Card
    st.markdown("""
    <div style="
        background-color: #FFFFFF; padding: 25px;
        border-radius: 16px; border: 2px solid #FFD68A;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        max-width: 350px; margin: auto;
    ">
        <h3 style="text-align:center; color:#FF7B00;">🔐 Login</h3>
    """, unsafe_allow_html=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("➡️ Start Practicing!", use_container_width=True):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("🎉 Welcome! Loading your session...")
            st.rerun()
        else:
            st.error("❌ Incorrect username or password!")

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


ensure_login()


# ============================================================================
# LOAD DATA
# ============================================================================

@st.cache_data
def load_data():
    base = DATA_DIR
    with open(f"{base}/data_mapped.json") as f:
        dat_map_json = json.load(f)
    with open(f"{base}/standards_mapping.json") as f:
        standards_mapping_json = json.load(f)
    with open(f"{base}/example_dat.json") as f:
        ex_dat = json.load(f)
    return dat_map_json, standards_mapping_json, ex_dat

dat_map_json, standards_mapping_json, ex_dat = load_data()


# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    
if 'mwp_session' not in st.session_state:
    with st.spinner("Initializing your personalized learning session..."):
        api_key = os.environ.get("OPENAI_API_KEY", "")
        st.session_state.mwp_session = MWPSession(
            api_key=api_key,
            user_id=st.session_state.username,
            output_dir=os.getcwd() + "/session_data"
        )

if 'submitted' not in st.session_state:
  st.session_state.submitted = False
if 'user_answer' not in st.session_state:
  st.session_state.user_answer = ""
if 'problem_queue' not in st.session_state:
  st.session_state.problem_queue = []
if 'current_problem_index' not in st.session_state:
  st.session_state.current_problem_index = 0
if 'total_correct' not in st.session_state:
  st.session_state.total_correct = 0
if 'total_attempted' not in st.session_state:
  st.session_state.total_attempted = 0
if 'next_button' not in st.session_state:
  st.session_state.next_button = False
if 'exit_button' not in st.session_state:
  st.session_state.exit_button = False
if 'background_generation_started' not in st.session_state:
  st.session_state.background_generation_started = False
if 'generation_complete' not in st.session_state:
  st.session_state.generation_complete = False
if 'bg_output_folder' not in st.session_state:
    st.session_state.bg_output_folder = None
if 'bg_process' not in st.session_state:
    st.session_state.bg_process = None
if 'bg_log' not in st.session_state:
    st.session_state.bg_log = []


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def choose_next_state(target_dl, target_text_l, n_total, n_corrects):
    text_l = ['low', 'medium', 'high']
    dl11_text_l = ['low', 'medium']

    if n_total == 0:
        return target_dl, target_text_l

    if ((n_corrects/n_total) >= 0.6):
        if target_dl < 11:
            nxt_dl, nxt_text = target_dl + 1, random.choice(text_l)
        elif target_dl == 11:
            nxt_dl, nxt_text = target_dl, random.choice(dl11_text_l)
    else:
        if target_dl == 1:
            nxt_dl = target_dl
            nxt_text = target_text_l if target_text_l=="low" else text_l[text_l.index(target_text_l)-1]
        else:
            if target_text_l=="low":
                nxt_dl, nxt_text = target_dl - 1, random.choice(text_l)
            else:
                nxt_dl, nxt_text = target_dl, text_l[text_l.index(target_text_l)-1]
    return nxt_dl, nxt_text


def generate_problems_background():
    if not st.session_state.background_generation_started:
        session_data = {
            "api_key": st.session_state.mwp_session.api_key,
            "curr_dl": st.session_state.mwp_session.curr_dl,
            "curr_text_l": st.session_state.mwp_session.curr_text_l,
            "ex_dat_path": "/data/example_dat.json",
            "standards_path": "/data/standards_mapping.json",
            "existing_queue": st.session_state.problem_queue
        }
        session_file = st.session_state.mwp_session.user_dir / "bg_session.json"
        with open(session_file, "w") as f:
            json.dump(session_data, f)

        output_folder = st.session_state.mwp_session.user_dir / 'bg_problems'
        output_folder.mkdir(exist_ok=True)

        import subprocess
        worker_script = 'background_worker.py'

        try:
            # add PYTHONPATH to ensure imports work
            env = os.environ.copy()
            env['PYTHONPATH'] = BASE_DIR

            # run as background process with error logging
            log_file = st.session_state.mwp_session.user_dir / 'bg_worker.log'

            with open(log_file, 'w') as log:
                process = subprocess.Popen(
                    [sys.executable, worker_script, str(session_file), str(output_folder)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env
                )

            st.session_state.background_generation_started = True
            st.session_state.generation_complete = False
            st.session_state.bg_output_folder = output_folder
            st.session_state.bg_process = process
            st.session_state.bg_log_file = log_file
            st.session_state.bg_log.append(f"✓ Launched background worker (PID: {process.pid})")
            st.session_state.bg_log.append(f"📝 Session file: {session_file}")
            st.session_state.bg_log.append(f"📂 Output folder: {output_folder}")
            st.session_state.bg_log.append(f"📄 Log file: {log_file}")

        except Exception as e:
            st.session_state.bg_log.append(f"❌ Failed to launch worker: {e}")
            st.error(f"Failed to launch background worker: {e}")


def check_background_completion():

    if st.session_state.bg_output_folder is None:
        return False

    output_folder = st.session_state.bg_output_folder

    # look for problem_*.json files
    if output_folder.exists():
        problem_files = sorted(output_folder.glob('problem_*.json'))

        if len(problem_files) > 0:
            st.session_state.bg_log.append(f"✓ Found {len(problem_files)} problem files")

            # load all problem files
            new_problems = []
            for pf in problem_files:
                try:
                    with open(pf, 'r') as f:
                        problem = json.load(f)
                        new_problems.append(problem)
                except Exception as e:
                    st.session_state.bg_log.append(f"❌ Error loading {pf.name}: {e}")

            if len(new_problems) > 0:
                # add new problems to queue
                remaining_needed = 5 - len(st.session_state.problem_queue)
                st.session_state.problem_queue.extend(new_problems[:remaining_needed])
                st.session_state.bg_log.append(f"✅ Loaded {len(new_problems)} problems from background worker")

                # clean up files
                import shutil
                try:
                    shutil.rmtree(output_folder)
                    st.session_state.bg_log.append("✓ Cleaned up output folder")
                except:
                    pass

                # clean up session file
                session_file = st.session_state.mwp_session.user_dir / 'bg_session.json'
                if session_file.exists():
                    session_file.unlink()
                    st.session_state.bg_log.append("✓ Cleaned up session file")

                st.session_state.generation_complete = True
                st.session_state.background_generation_started = False
                st.session_state.bg_output_folder = None
                return True

    return False


def generate_initial_problems():
    while len(st.session_state.problem_queue) <  1:
        problem_cand = st.session_state.mwp_session.generator.generate_item(
            ex_dat = ex_dat,
            standards_dat = standards_mapping_json,
            n_examples = 2,
            n_items = 5,
            model_option = 'gpt-4.1-mini',
            rationale_dat = True)
        new_problems = st.session_state.mwp_session.generator.evaluator.parallel_evaluation(problem_cand)

        if (len(new_problems) > 0):
            st.session_state.problem_queue = new_problems #st.session_state.mwp_session.generator.problem_db

    return len(st.session_state.problem_queue) <  5


# check answer
def check_answer(user_answer, correct_answer):
    try:
        return int(user_answer) == int(correct_answer)
    except:
        return False


# save data
def save_session_question_details():
    if "mwp_session" not in st.session_state or "username" not in st.session_state:
        return

    username = st.session_state.username
    mwp_session = st.session_state.mwp_session

    if not hasattr(mwp_session, "curr_session_data") or len(mwp_session.curr_session_data) == 0:
        return

    session_end = datetime.now().isoformat(timespec="seconds")

    records = []
    for idx, prob in enumerate(mwp_session.curr_session_data):
        records.append({
            "username": username,
            "session_end": session_end,
            "question_index": idx + 1,
            "math_dl": prob.get("math_dl"),
            "math_text_l": prob.get("math_text_l"),
            "math_standard": standards_mapping_json[str(prob.get("math_dl"))]["Extended Standards"],
            "score": prob.get("Score")
        })

    df = pd.DataFrame(records)

    user_dir = mwp_session.user_dir
    os.makedirs(user_dir, exist_ok=True)
    details_path = user_dir / "question_history.csv"

    if details_path.exists():
        df_existing = pd.read_csv(details_path)
        df_all = pd.concat([df_existing, df], ignore_index=True)
    else:
        df_all = df

    df_all.to_csv(details_path, index=False)

def load_question_history_for_user():
    if "mwp_session" not in st.session_state or "username" not in st.session_state:
        return None

    user_dir = st.session_state.mwp_session.user_dir
    path = user_dir / "question_history.csv"

    if not path.exists():
        st.info("No question history found yet for this user.")
        return None

    df = pd.read_csv(path)

    df["session_end"] = pd.to_datetime(df["session_end"])

    return df


# ============================================================================
# GENERATE INITIAL PROBLEMS
# ============================================================================

if len(st.session_state.problem_queue) == 0:
    generate_initial_problems()

# transition to next math difficulty level or text complexity level
if st.session_state.current_problem_index >= len(st.session_state.problem_queue):
    st.session_state.mwp_session.curr_dl, st.session_state.mwp_session.curr_text_l = choose_next_state(
        st.session_state.mwp_session.curr_dl,
        st.session_state.mwp_session.curr_text_l,
        st.session_state.total_attempted,
        st.session_state.total_correct
    )

    st.session_state.mwp_session.generator = MWPGenerator(
    api_key=st.session_state.mwp_session.api_key,
    target_dl=st.session_state.mwp_session.curr_dl,
    target_text_l=st.session_state.mwp_session.curr_text_l)

    st.session_state.problem_queue = []
    st.session_state.current_problem_index = 0
    st.session_state.total_correct = 0
    st.session_state.total_attempted = 0
    generate_initial_problems()


# =========================
# SIDEBAR: PROGRESS TRACKER
# =========================

st.sidebar.header("📊 Your Progress")
st.sidebar.metric("Math Difficulty Level", st.session_state.mwp_session.curr_dl)
st.sidebar.metric("Text Complexity Level", st.session_state.mwp_session.curr_text_l)
st.sidebar.metric("Problems Attempted", st.session_state.total_attempted)
st.sidebar.metric("Correct Answers", st.session_state.total_correct)

if st.session_state.total_attempted > 0:
    accuracy = (st.session_state.total_correct / st.session_state.total_attempted) * 100
    st.sidebar.metric("Accuracy", f"{accuracy:.1f}%")
    st.sidebar.progress(accuracy / 100)

st.sidebar.markdown("---")
st.sidebar.write(f"**Current Problem:** {st.session_state.current_problem_index + 1} of {len(st.session_state.problem_queue)}")
st.sidebar.write(f"**Total Problems for the Session:** {len(st.session_state.mwp_session.curr_session_data)}")

if len(st.session_state.problem_queue) > 0 and len(st.session_state.problem_queue) < 5:
    generate_problems_background()

# show generation status
if st.session_state.background_generation_started and not st.session_state.generation_complete:
    st.sidebar.info(f"🔄 Background worker processing... ({len(st.session_state.problem_queue)}/5 ready)")

    with st.sidebar.expander("🔍 Generation Log"):
        if len(st.session_state.bg_log) > 0:
            for line in st.session_state.bg_log[-12:]:
                st.write(line)
        else:
            st.write("Waiting for background worker...")

elif len(st.session_state.problem_queue) >= 5:
    st.sidebar.success("✅ All 5 problems ready!")


# ============================================================================
# MAIN CONTENT: DISPLAY PROBLEM
# ============================================================================

# catch the problem to be displayed
current_problem = st.session_state.problem_queue[st.session_state.current_problem_index] #get_current_problem()

if not st.session_state.exit_button:
    st.header("❓ Your Challenge!")
    st.write(current_problem['Q'])

    st.subheader("🔢 Your Answer:")
    user_answer = st.text_input(
        "Enter your Answer: ",
        value=st.session_state.user_answer,
        disabled=st.session_state.submitted
    )

    if not st.session_state.submitted:
        st.session_state.user_answer = user_answer
        if st.button("Submit Your Answer"):
            completed = check_background_completion()
            try:
                user_answer = int(user_answer)
                st.session_state.submitted = True
                st.session_state.total_attempted += 1
                is_correct = check_answer(st.session_state.user_answer, current_problem['A'])
                if is_correct:
                    st.session_state.total_correct += 1
                st.rerun()
            except ValueError:
                st.error("Please enter a valid number.")

    if st.session_state.submitted:

        is_correct = check_answer(st.session_state.user_answer, current_problem['A'])

        if is_correct:
            current_problem['Score'] = 1
            st.balloons()
            st.success("🎉 Correct! Great job! 🎉")
        else:
            current_problem['Score'] = 0
            st.error("❌ Good try! Here's how to solve it:")

        current_problem['math_dl'] = st.session_state.mwp_session.curr_dl
        current_problem['math_text_l'] = st.session_state.mwp_session.curr_text_l

        st.markdown("---")
        st.subheader("📖 Explanation")
        st.write(current_problem['R'])
        st.subheader("🧮 Helpful Formula")
        st.code(current_problem['F'])
        st.subheader("✅ Correct Answer")
        st.code(str(current_problem['A']))

        st.markdown("---")
        if st.button("➡️ Next Question"):
            completed = check_background_completion()
            st.session_state.current_problem_index += 1
            st.session_state.mwp_session.curr_session_data.append(current_problem)
            st.session_state.submitted = False
            st.session_state.user_answer = ""
            st.rerun()

    if st.button("Exit"):
        if st.session_state.submitted == False:
            current_problem['Score'] = 0
            current_problem['math_dl'] = st.session_state.mwp_session.curr_dl
            current_problem['math_text_l'] = st.session_state.mwp_session.curr_text_l
        st.session_state.mwp_session.curr_session_data.append(current_problem)
        st.session_state.exit_button = True
        st.session_state.mwp_session.save_data()
        save_session_question_details()
        st.rerun()



# ============================================================================
# SUMMARY PAGE
# ============================================================================

if st.session_state.exit_button:
    st.header("🌟 Great Job Today!")

    df_hist = load_question_history_for_user()

    if df_hist is not None:
        n_attempted = st.session_state.total_attempted
        n_correct = st.session_state.total_correct
        last_dl = st.session_state.mwp_session.curr_dl
        accuracy = (n_correct / n_attempted * 100) if n_attempted > 0 else 0.0

        st.subheader("🏆 Today's Math Practice")
        st.write(f"You worked on **{n_attempted} problems** today, and you got **{n_correct} correct**!")
        st.write(f"You got **{accuracy:.1f}%** of your problems correct today!")

        st.subheader("🎯 Today's Challenge Level")
        st.write(f"**Math Level:** {last_dl}")
        st.write("Great job working through today's challenge!")


        st.markdown("---")
        st.subheader("📚 Your Practice So Far")
        total_accuracy = df_hist["score"].mean() * 100
        num_sessions = df_hist["session_end"].nunique()
        total_questions = len(df_hist)
        total_correct_all = df_hist["score"].sum()

        st.write(f"So far, you have practiced on **{num_sessions} different days**.")
        st.write(f"In total, you have worked on **{total_questions} problems** and got **{total_correct_all} right**!")
        st.write(f"That means you got **{total_accuracy:.1f}%** of all your problems correct so far.")

        st.markdown("---")
        st.header("📌 Current Session Overview (for parents/teachers)")

        last_session_time = df_hist["session_end"].max()
        df_last = df_hist[df_hist["session_end"] == last_session_time]

        last_acc = df_last["score"].mean() * 100

        st.metric("Accuracy (Most Recent Session)", f"{last_acc:.1f}%")

        acc_last_by_dl = (
            df_last
            .groupby("math_dl")["score"]
            .mean()
            .reset_index()
            .rename(columns={"score": "accuracy"})
        )
        acc_last_by_dl["accuracy"] = acc_last_by_dl["accuracy"] * 100

        st.write("Accuracy by Difficulty Level (Most Recent Session)")

        chart_last_dl = (
            alt.Chart(acc_last_by_dl)
            .mark_bar(color="#ffcfd2")
            .encode(
                x=alt.X("math_dl:O", title="Math Difficulty Level", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("accuracy:Q", title="Accuracy (%)", scale=alt.Scale(domain=[0,100])),
                tooltip=["math_dl", "accuracy"]
            )
        )
        st.altair_chart(chart_last_dl, use_container_width=True)

        st.header("📈 All Sessions Overview (for parents/teachers)")
        st.metric("Total Accuracy", f"{total_accuracy:.1f}%")

        acc_by_dl = (
            df_hist
            .groupby("math_dl")["score"]
            .mean()
            .reset_index()
            .rename(columns={"score": "accuracy"})
        )
        acc_by_dl["accuracy"] = acc_by_dl["accuracy"] * 100

        st.write("Accuracy by Difficulty Level (All Sessions)")

        chart_dl = (
            alt.Chart(acc_by_dl)
            .mark_bar(color="#f1c0e8")
            .encode(
                x=alt.X("math_dl:O", title="Math Difficulty Level", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("accuracy:Q", title="Accuracy (%)", scale=alt.Scale(domain=[0,100])),
                tooltip=["math_dl", "accuracy"]
            )
        )
        st.altair_chart(chart_dl, use_container_width=True)

        acc_by_session = (
            df_hist
            .groupby("session_end")["score"]
            .mean()
            .reset_index()
            .sort_values("session_end")
            .rename(columns={"score": "accuracy"})
        )
        acc_by_session["accuracy"] = acc_by_session["accuracy"] * 100

        st.write("Accuracy by Session")

        chart_session = (
            alt.Chart(acc_by_session)
            .mark_line(point=True, color="#cfbaf0")
            .encode(
                x=alt.X("session_end:T", title="Session End Time"),
                y=alt.Y("accuracy:Q", title="Accuracy (%)", scale=alt.Scale(domain=[0,100])),
                tooltip=["session_end", "accuracy"]
            )
        )
        st.altair_chart(chart_session, use_container_width=True)

        st.write("Number of Problems Attempted per Session")

        problems_per_session = (
            df_hist
            .groupby("session_end")
            .size()
            .reset_index(name="num_problems")
            .sort_values("session_end")
        )

        chart_problems = (
            alt.Chart(problems_per_session)
            .mark_bar(color="#a3c4f3")
            .encode(
                x=alt.X("session_end:T", title="Session End Time"),
                y=alt.Y("num_problems:Q", title="Number of Problems"),
                tooltip=["session_end", "num_problems"]
            )
        )
        st.altair_chart(chart_problems, use_container_width=True)

