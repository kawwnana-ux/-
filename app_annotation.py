import io
import re
import json
import zipfile
from datetime import datetime

import pandas as pd
import streamlit as st

try:
    import spacy
except ImportError:
    spacy = None


st.set_page_config(
    page_title="特許SAO 教師データ作成",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 特許SAO 教師データ作成ツール")
st.caption("GiNZAでトークン化 → 構成要素をBIOアノテーション → 構成要素間のSAO関係をアノテーション → CSV保存")


# ============================================================
# 設定
# ============================================================

LABELS = ["O", "B-COMP", "I-COMP"]

RELATION_TYPES = [
    "有する",
    "含む",
    "配置",
    "位置決めされる",
    "間に位置する",
    "接続",
    "設ける",
    "形成",
    "固定",
    "装着",
    "収容",
    "備える",
    "その他",
]

GENERIC_NON_COMPONENTS = {
    "前記",
    "該",
    "少なく",
    "少なくとも",
    "少なくとも１つの",
    "少なくとも1つの",
    "一部",
}


# ============================================================
# GiNZA
# ============================================================

@st.cache_resource
def load_ginza():
    if spacy is None:
        raise RuntimeError("spaCy がインストールされていません。")
    for model_name in ("ja_ginza", "ja_ginza_electra"):
        try:
            return spacy.load(model_name)
        except Exception:
            pass
    raise RuntimeError(
        "GiNZAモデルを読み込めませんでした。"
        "requirements.txt に ja-ginza と ginza が入っているか確認してください。"
    )


def tokenize_claim(text):
    nlp = load_ginza()
    doc = nlp(text)
    rows = []

    for i, tok in enumerate(doc):
        if not tok.text.strip():
            continue

        rows.append({
            "token_id": i,
            "token": tok.text,
            "pos": tok.pos_,
            "lemma": tok.lemma_,
            "dep": tok.dep_,
            "head": tok.head.i,
            "label": "O",
        })

    return pd.DataFrame(rows)


def get_component_spans(token_df):
    """
    BIOラベルから連続する構成要素を抽出。
    token_id と表示番号を使って、同じ構成要素名が複数存在しても区別する。
    """
    spans = []
    current = []

    for _, row in token_df.iterrows():
        label = row["label"]

        if label == "B-COMP":
            if current:
                spans.append(current)
            current = [row]
        elif label == "I-COMP":
            if current:
                current.append(row)
            else:
                # Iから始まった場合は、教師データ作成画面で扱いやすいよう
                # 暫定的にBとして開始
                current = [row]
        else:
            if current:
                spans.append(current)
                current = []

    if current:
        spans.append(current)

    components = []
    for idx, span in enumerate(spans, start=1):
        components.append({
            "component_id": idx,
            "text": "".join(str(x["token"]) for x in span),
            "start_token": int(span[0]["token_id"]),
            "end_token": int(span[-1]["token_id"]),
        })

    return pd.DataFrame(components)


def normalize_bio(df):
    """
    I-COMPの直前が構成要素でない場合はB-COMPに直す。
    ユーザーがO→I-COMPとしてしまった場合の教師データ破損を防止。
    """
    df = df.copy()
    previous_is_component = False

    for i in df.index:
        label = df.at[i, "label"]

        if label == "I-COMP" and not previous_is_component:
            df.at[i, "label"] = "B-COMP"
            previous_is_component = True
        elif label == "B-COMP":
            previous_is_component = True
        elif label == "I-COMP":
            previous_is_component = True
        else:
            previous_is_component = False

    return df


def make_component_training_csv(token_df, claim_id, claim_text):
    out = token_df.copy()
    out.insert(0, "claim_id", claim_id)
    out["claim_text"] = claim_text

    cols = [
        "claim_id",
        "token_id",
        "token",
        "label",
        "pos",
        "lemma",
        "dep",
        "head",
        "claim_text",
    ]
    return out[cols]


def make_relation_training_csv(relations, claim_id, claim_text):
    if not relations:
        return pd.DataFrame(
            columns=[
                "claim_id",
                "source",
                "relation",
                "target",
                "source_id",
                "target_id",
                "claim_text",
            ]
        )

    out = pd.DataFrame(relations)
    out.insert(0, "claim_id", claim_id)
    out["claim_text"] = claim_text

    cols = [
        "claim_id",
        "source",
        "relation",
        "target",
        "source_id",
        "target_id",
        "claim_text",
    ]
    return out[cols]


def make_zip(component_df, relation_df):
    buf = io.BytesIO()

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(
            "components.csv",
            component_df.to_csv(index=False).encode("utf-8-sig"),
        )
        z.writestr(
            "relations.csv",
            relation_df.to_csv(index=False).encode("utf-8-sig"),
        )

    buf.seek(0)
    return buf.getvalue()


# ============================================================
# Session state
# ============================================================

defaults = {
    "claim_id": "001",
    "claim_text": "",
    "token_df": None,
    "relations": [],
    "history_components": [],
    "history_relations": [],
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================
# サイドバー
# ============================================================

with st.sidebar:
    st.header("📌 アノテーション規則")

    st.markdown(
        """
**構成要素（COMPONENT）**
- 装置・部品・モジュール
- 端子
- 発明を構成する機能的構成

**O（構成要素ではない）**
- 前記
- 少なくとも
- 少なくとも１つの
- 主面
- 一部
- 数量表現
- 助詞・動詞など

**関係**
- 有する
- 含む
- 配置
- 位置決めされる
- 間に位置する
- その他
"""
    )

    st.divider()
    st.markdown("### 今回の重要ルール")
    st.info(
        "「主面」は原則として独立した構成要素にしません。\n\n"
        "「少なく」も構成要素にしません。\n\n"
        "「前記」は既出構成要素への照応であり、新しい構成要素ではありません。"
    )


# ============================================================
# 1. 請求項入力
# ============================================================

st.header("1️⃣ 請求項を入力")

c1, c2 = st.columns([1, 5])

with c1:
    claim_id = st.text_input(
        "請求項ID",
        value=st.session_state.claim_id,
        help="例：001、SEMICON-001",
    )

with c2:
    claim_text = st.text_area(
        "請求項本文",
        value=st.session_state.claim_text,
        height=220,
        placeholder="ここに特許請求項を貼り付けてください。",
    )

if st.button("🔍 GiNZAでトークン化", type="primary"):
    if not claim_text.strip():
        st.warning("請求項本文を入力してください。")
    else:
        try:
            df = tokenize_claim(claim_text)
            st.session_state.claim_id = claim_id.strip() or "001"
            st.session_state.claim_text = claim_text
            st.session_state.token_df = df
            st.session_state.relations = []
            st.success(f"{len(df)}トークンに分割しました。")
        except Exception as e:
            st.error(f"GiNZAの読み込み・解析に失敗しました: {e}")


# ============================================================
# 2. 構成要素アノテーション
# ============================================================

if st.session_state.token_df is not None:
    st.divider()
    st.header("2️⃣ 構成要素をアノテーション")

    st.markdown(
        """
各トークンのラベルを変更してください。

- **B-COMP**：構成要素の開始
- **I-COMP**：同じ構成要素の続き
- **O**：構成要素ではない

例：`パワー 半導体 モジュール` → `B-COMP / I-COMP / I-COMP`
"""
    )

    token_df = st.session_state.token_df.copy()

    edited = st.data_editor(
        token_df,
        hide_index=True,
        width="stretch",
        num_rows="fixed",
        column_config={
            "token_id": st.column_config.NumberColumn(
                "ID", disabled=True, width="small"
            ),
            "token": st.column_config.TextColumn(
                "トークン", disabled=True
            ),
            "pos": st.column_config.TextColumn(
                "品詞", disabled=True, width="small"
            ),
            "lemma": st.column_config.TextColumn(
                "原形", disabled=True
            ),
            "dep": st.column_config.TextColumn(
                "係り受け", disabled=True
            ),
            "head": st.column_config.NumberColumn(
                "head", disabled=True, width="small"
            ),
            "label": st.column_config.SelectboxColumn(
                "構成要素ラベル",
                options=LABELS,
                required=True,
            ),
        },
        key="component_editor",
    )

    if st.button("💾 構成要素ラベルを確定"):
        edited = normalize_bio(edited)
        st.session_state.token_df = edited

        components = get_component_spans(edited)

        if len(components) == 0:
            st.warning("構成要素が1つもありません。ラベルを確認してください。")
        else:
            st.success(f"{len(components)}個の構成要素を認識しました。")

    # 現在の構成要素
    components = get_component_spans(st.session_state.token_df)

    if not components.empty:
        st.subheader("現在の構成要素")

        display_components = components.copy()
        display_components["表示"] = display_components.apply(
            lambda r: f'{int(r["component_id"])}. {r["text"]}',
            axis=1,
        )
        st.dataframe(
            display_components[["表示", "start_token", "end_token"]],
            hide_index=True,
            width="stretch",
        )

        # ====================================================
        # 3. 関係アノテーション
        # ====================================================

        st.divider()
        st.header("3️⃣ 構成要素間の関係をアノテーション")

        st.markdown(
            """
**source → relation → target** の順で指定してください。

例：

`パワー半導体モジュール → 含む → 出力端子`
"""
        )

        component_options = {
            f'{int(row.component_id)}. {row.text}': int(row.component_id)
            for _, row in components.iterrows()
        }

        option_names = list(component_options.keys())

        with st.form("relation_form", clear_on_submit=True):
            r1, r2, r3 = st.columns([3, 2, 3])

            with r1:
                source_name = st.selectbox(
                    "主語 / source",
                    option_names,
                    key="source_select",
                )

            with r2:
                relation = st.selectbox(
                    "関係",
                    RELATION_TYPES,
                    key="relation_select",
                )

            with r3:
                target_name = st.selectbox(
                    "目的語 / target",
                    option_names,
                    key="target_select",
                )

            submitted = st.form_submit_button(
                "＋ この関係を追加",
                type="primary",
            )

        if submitted:
            source_id = component_options[source_name]
            target_id = component_options[target_name]

            if source_id == target_id:
                st.warning("sourceとtargetは別の構成要素を選択してください。")
            else:
                source_text = components.loc[
                    components["component_id"] == source_id, "text"
                ].iloc[0]

                target_text = components.loc[
                    components["component_id"] == target_id, "text"
                ].iloc[0]

                new_relation = {
                    "source": source_text,
                    "relation": relation,
                    "target": target_text,
                    "source_id": source_id,
                    "target_id": target_id,
                }

                # 同一関係の重複防止
                duplicate = any(
                    r["source_id"] == source_id
                    and r["relation"] == relation
                    and r["target_id"] == target_id
                    for r in st.session_state.relations
                )

                if duplicate:
                    st.warning("同じ関係はすでに登録されています。")
                else:
                    st.session_state.relations.append(new_relation)
                    st.success(
                        f"{source_text} → {relation} → {target_text}"
                    )

        # 現在の関係
        st.subheader("現在の関係")

        if st.session_state.relations:
            rel_df = pd.DataFrame(st.session_state.relations)

            view = rel_df.copy()
            view.insert(
                0,
                "No.",
                range(1, len(view) + 1),
            )

            st.dataframe(
                view[
                    [
                        "No.",
                        "source",
                        "relation",
                        "target",
                    ]
                ],
                hide_index=True,
                width="stretch",
            )

            delete_no = st.number_input(
                "削除する関係No.",
                min_value=1,
                max_value=len(st.session_state.relations),
                value=1,
                step=1,
            )

            if st.button("🗑️ 選択した関係を削除"):
                st.session_state.relations.pop(int(delete_no) - 1)
                st.rerun()
        else:
            st.info("まだ関係が登録されていません。")

        # ====================================================
        # 4. 保存
        # ====================================================

        st.divider()
        st.header("4️⃣ 教師データとして保存")

        final_token_df = normalize_bio(st.session_state.token_df)

        component_csv = make_component_training_csv(
            final_token_df,
            st.session_state.claim_id,
            st.session_state.claim_text,
        )

        relation_csv = make_relation_training_csv(
            st.session_state.relations,
            st.session_state.claim_id,
            st.session_state.claim_text,
        )

        st.write(
            f"構成要素ラベル：{len(component_csv)}トークン  /  "
            f"関係：{len(relation_csv)}件"
        )

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.download_button(
                "⬇️ components.csv",
                component_csv.to_csv(index=False).encode("utf-8-sig"),
                file_name="components.csv",
                mime="text/csv",
            )

        with col_b:
            st.download_button(
                "⬇️ relations.csv",
                relation_csv.to_csv(index=False).encode("utf-8-sig"),
                file_name="relations.csv",
                mime="text/csv",
            )

        with col_c:
            zip_bytes = make_zip(component_csv, relation_csv)
            st.download_button(
                "📦 2ファイルをZIPで保存",
                zip_bytes,
                file_name="patent_sao_training_data.zip",
                mime="application/zip",
                type="primary",
            )

        # ====================================================
        # 5. 次の請求項
        # ====================================================

        st.divider()
        st.header("5️⃣ 次の請求項へ")

        st.warning(
            "次の請求項へ進む前に、上のZIPまたはCSVを保存してください。"
        )

        if st.button("➡️ アノテーションをリセットして次へ"):
            st.session_state.claim_id = ""
            st.session_state.claim_text = ""
            st.session_state.token_df = None
            st.session_state.relations = []
            st.rerun()

else:
    st.info("まず請求項を入力して「GiNZAでトークン化」を押してください。")


# ============================================================
# フッター
# ============================================================

st.divider()
st.caption(
    "教師データ作成の目的：GiNZAの解析結果を正解データにするのではなく、"
    "人間がアノテーションした正解データを作り、後で機械学習モデルの学習・評価に利用する。"
)
