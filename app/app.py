import sys
from pathlib import Path
import streamlit as st

# 1. Project configuration

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_pipeline import answer_question, document_count


# 2. Streamlit page configuration

st.set_page_config(
    page_title="Industrial Maintenance Copilot",
    page_icon="⚙️",
    layout="wide",
)


# 3. Application header

st.title("⚙️ Industrial Maintenance Copilot")

st.markdown(
    """
    Ask questions about your equipment maintenance manuals.

    The application retrieves relevant manual excerpts and
    uses an LLM to generate an answer grounded in those excerpts.
    """
)

st.caption(
    f"Knowledge base: {document_count} stored document chunks"
)

st.divider()


# 4. Question input form

with st.form("maintenance_question_form"):
    question = st.text_area(
        "Enter your maintenance question",
        placeholder="Example: What causes bearing overheating?",
        height=100,
    )

    submitted = st.form_submit_button(
        "Get Answer",
        type="primary",
        use_container_width=True,
    )


# 5. Generate and display the answer

if submitted:

    if not question.strip():
        st.warning("Please enter a question first.")

    else:
        try:
            with st.spinner(
                "Searching manuals and generating an answer..."
            ):
                result = answer_question(
                    query=question.strip(),
                    top_k=3,
                )

            st.subheader("Answer")
            st.markdown(result["answer"])

            st.subheader("Retrieved Sources")

            if result["sources"]:

                for index, source in enumerate(
                    result["sources"],
                    start=1,
                ):
                    filename = source["source"]
                    page = source["page"]

                    with st.expander(
                        f"{index}. {filename} — Page {page}"
                    ):
                        st.markdown(
                            f"**Chunk index:** "
                            f"{source['chunk_index']}"
                        )

                        st.markdown("**Retrieved excerpt:**")
                        st.write(source["text"])

                        st.caption(
                            f"Retrieval distance: "
                            f"{source['distance']:.4f}"
                        )

            else:
                st.info(
                    "No source excerpts were retrieved."
                )

        except Exception as e:
            st.error(
                "Something went wrong while answering your question."
            )
            st.exception(e)