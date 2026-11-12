# EXPERIMENTS：實驗規格書 v3（分析計畫凍結版）

本文件是**所有正式實驗的唯一依據**，內容包括：要測什麼、怎麼測、量哪些數據、用哪些資料、怎麼分析，以及結果怎麼解讀。
- **凍結程序（W02）**：在跑任何 held-out 模型工作之前，先 commit 下列全部內容，確認新檔確實進入 commit（不能有 untracked 的核心檔），再打 git tag `prereg-v3`：
  1. 本規格書（`EXPERIMENTS.md`）與 `psych101/PROTOCOL.md`；
  2. 工作清單 manifest：`portable/make_jobs.py`、`portable/jobs.jsonl`、`portable/models.json`、`portable/run_jobs.py`、`portable/setup_data.sh`；
  3. 所有 runner：`run_volatile.py`、`run_controls.py`、`run_thinking.py`、`psych101/run_xiong.py`，以及它們 import 的 `../source_trust/model_backends.py`、`backend_multi.py`、`controller.py`、`scenarios.py`、`generate_episodes.py`、`generate_robust.py`、`psych101/render_xiong.py`；
  4. 所有分析程式：`analyze_main.py`、`analyze_mechanism.py`、`analyze_controls.py`、`analyze_thinking.py`（已實作並納入 `prereg-v3`）、`cogmodels.py`、`design_sim_runner_rule.py`、`psych101/analyze_xiong.py`，以及對應的 `test_*.py`；
  5. 模型版本：`portable/models.json` 中每個模型固定的 HF revision；
  6. 資料 checksum：`portable/checksums.json`（episodes_v3、episodes_robust_v3、image index、Psych-101 切片的 sha256）。
  驗收：從 tag 重新 checkout，能重建相同的 jobs.jsonl 與相同的資料 hash。
- **凍結前已經看過的東西**（照實揭露）：
  - 3090 前導結果，資料是 v1（§6）；
  - v3 dev 上只有 E1 對照的 Qwen3-VL-8B 執行，而且只看過部分結果；
  - held-out（與 robust）上的規則基準（§2.4），只是資料生成器的性質，不含任何模型輸出；
  - 真實模型的 smoke test：在少數 held-out 題目上只為了檢查管線（載入、讀出、續跑），輸出沒有做任何分析。
- 措辭：論文寫「分析計畫在 held-out 評估之前固定（fixed before held-out evaluation；時間戳記 commit／tag）」。**我們不做正式的預先登錄（例如 OSF），也不宣稱 preregistered**；tag 名稱中的 `prereg` 只是標籤。之後若有任何改動，都會記在 §7，並標明是在看到 held-out 模型結果之前還是之後。
- 舊版（v1／v2 的規格與 `DESIGN.md`）已被本文件取代。
- 工作清單：`portable/make_jobs.py` → `portable/jobs.jsonl`（原 MI300X）；`--platform rtx3090` → `portable/jobs_3090.jsonl`（偏離 D2）。執行方式：`portable/README_RUN.md`。
- 分析程式：`analyze_main.py`、`analyze_mechanism.py`、`analyze_controls.py`、`analyze_thinking.py`（E9，已納入凍結 tag）、`cogmodels.py`、`psych101/analyze_xiong.py`。

---

## 0. 研究問題、主張與主要檢定

**研究問題**：凍結的 VLM 能不能從 in-context 的回饋經驗中學會「該信哪個來源」？我們把這件事拆成三步：① 從紀錄中判斷每回合誰對；② 跨回合累加；③ 使用累加的結果。如果做不到，是卡在哪一步？

**理論框架**：
- **描述—經驗落差**（description–experience gap；Hertwig & Erev 2004）：人「被告知機率」和「從經驗中學到機率」時，會做出不同的選擇。
- **波動環境中的社會性學習**：Behrens et al. 2008。

**主張**（每一條都要由實驗支持，不成立就不寫）：

| 主張 | 實驗 |
|---|---|
| **C1（主要）** 原始紀錄—摘要落差（raw vs. summarized history gap，C−A）：檢定同一段歷史以計數摘要（C，summarized history）呈現時的準確率，是否高於以原始逐回合紀錄（A）呈現時。分開報告：(i) 配對落差 C−A；(ii) A 和 C 各自的絕對表現（「A 接近亂猜」只有在 A 落在事先定義的等價界限 0.5 ± 0.03 內才能宣稱）；(iii) 在哪些模型／條件成立，例外照實保留。C 不是描述條件，C−A 不稱為描述—經驗落差。**D−A**（D：直接告知可靠度，才是描述條件；即描述—經驗落差／DH gap）由 E1 的對照題**另外報告**，用自己的指標，不與 C−A 放在同一縱軸或相減。不宣稱「所有 8B 以上的模型」，也不宣稱有能力門檻 | C−A：E2、E3；D−A：E1 |
| **C1b** 落差卡在哪一步：比較「原始紀錄」、「每回合標明誰對」（文字 B，或格子圖 B′）和「計數摘要」 | E1、E2 |
| **C1c** 行為描述（不是機制）：用捷思吻合度（κ，例如抄上一回合結果、跟隨上一回合說對者、全計數、BOCPD）和積分核，描述模型在原始紀錄與摘要下的選擇與哪些策略一致。只寫「與某策略一致」；C 條件的積分核反映外部摘要與模型兩者的合成效果 | E2（不需新增 GPU） |
| **C2** 把累加交給外部規則（全計數、固定視窗、BOCPD）後，檢定加入 VLM 的準確率和規則單獨預測的差距是否落在預設的實質界限（±2 個百分點，TOST）內。另外報告逐題選擇一致率。「表現等價」、「行為近似」、「內部機制相同」三者分開；第三者不由行為資料推出。規則之間的差異只在測試過的設定內描述（held-out，§2.4，用 runner 實際規則計算）：最佳固定視窗（K=5，.692）比 BOCPD（.678）約高 1.4 個百分點；全計數（.632）比 K=5 低約 6.0 個百分點、比 BOCPD 低約 4.6 個百分點 | E3、E8 |
| **C3（探索性）** 照片報告的綁定：在已告知可靠度時，檢查模型能否把照片報告綁定到對的報告者（例如是否預設回答第一張圖）；分模型報告，不推廣成普遍現象 | E4 |
| **C4（輔助、描述性）** 外部任務和人類比較：在 Psych-101 xiong2023 中，描述模型和人類的學習率隨 hazard 的變化；報告估計值、CI 與校正結果，不做人機差異或相似的強主張（H02，見 E6） | E6 |
| **C5** 推理模式：允許思考（Thinking／CoT）時，上述落差是否縮小 | E9 |
| **C6（參考）** API 頂尖模型 | E7（主任務定案後才跑） |

**不主張**：類腦機制、神經同源、與人類對齊、VLM 內在學會了變化點推論、首創 LLM 反轉學習或描述—經驗落差。前作見 `discussion/novelty_check_v2.md`。

### 唯一的主要檢定（held-out）
**原始紀錄—摘要落差（raw vs. summarized history gap）= acc(C：`sum_all` 全計數摘要) − acc(A：原始完整歷史)**

術語說明（依 `discussion/paper_conventions.md`）：
- C 是「摘要後的經驗」（summarized history，Krishnamurthy et al. 2024 的用語），不是「描述」。
- 真正的「描述」條件是 D：直接告知可靠度（E1 的 `explicit_rel`）。
- 「描述—經驗落差」（D 對 A）在 LLM 文獻中稱為 description–history（DH）gap（Ge et al. 2026），由 E1 報告。
- 條件：文字格式，分別在每個核心模型上計算。
- 以 episode 為單位配對，孿生版先平均。用 Wilcoxon signed-rank（雙尾）檢定。
- 多個核心模型之間用 Holm 校正。
- 判定標準：至少兩個不同語言模型家族（Qwen 與 Gemma）的落差顯著大於 0。
- **為什麼用 `sum_all` 當 C**：C 由**同一段歷史**計算，不引入額外觀察，也不做額外推論（不像 sum_cp 要推斷變化點）。但它**不是無損的**：計數丟掉了時間順序，在可靠度會反轉的任務中，不是預測當下可靠度的充分統計量。因此 C−A 支持「呈現方式／計算負擔很重要」，**不能**唯一歸因於「累加」這一步；要定位到哪一步，需搭配 B、A+ 與單回合判斷（E1 的 single_trial_who）。

其他比較都標為次要或探索性。

---

## 1. 共用設定

### 1.1 模型（`portable/models.json`，版本固定）
| key | 模型 | 語言模型家族 | 角色 |
|---|---|---|---|
| qwen3vl8b、qwen3vl32b | Qwen3-VL 8B／32B Instruct | Qwen3 | 核心 |
| internvl8b、internvl38b | InternVL3.5 8B／38B | Qwen3（與上同家族，只當作不同的視覺管線） | 核心 |
| **gemma3_27b** | Gemma-3-27B-it | **Gemma**（獨立家族） | 核心，需要先在 HF 同意授權 |
| qwen25vl72b | Qwen2.5-VL-72B | Qwen2.5 | 選配規模延伸；不在 3090 預設清單，待 H200 等算力可用 |
| qwen3vl8b_thinking、qwen3vl32b_thinking | Qwen3-VL Thinking | Qwen3 | 只用在 E9 |
| gemma3_12b、qwen3vl235b、internvl241b | — | — | 選配 |

- 平台：原規劃使用 MI300X；2026-11-12 因算力可用性改為先準備 RTX 3090 主力版本（偏離 D2，見 §7）。**之後已由偏離 D5 取代：正式評估改在 H200**（§7）。3090 執行計畫已於 2026-11-13 取消（§7）。同一個模型的正式條件仍在同一個平台完成，維持 BF16 主體與 FP32 輸出頭。3090 輸出使用獨立的 `runs_3090/`，保留 `runs_mi300x/` 與原清單。這項配置變更不代表跨平台驗證或所有模型的顯存測試已完成。
- 既有 3B／4B 的 3090 前導資料是 v1 版（有開頭介紹順序的混淆），論文中只放附錄，不作主張。新增的 4B v3 dev 計畫另見 §7，與這批 v1 資料分開，不加入原正式 held-out 檢定。

### 1.2 讀出方式
- **選擇題**：讀第一個 token 的原始 logit，在兩個候選答案之間 softmax，得到 y = P(預測 A 的報告)。強制 greedy，另外記錄 argmax。
- **讀出的解讀（R10）**：y 是「兩個答案選項之間的條件偏好（只在兩個候選 token 上正規化）」，不等於模型自由生成的選擇機率，也不是信念。每個模型 × 格式 × 條件都報告候選 token 的機率質量（平均、最小值、質量 < 0.5 的比例）和 greedy 輸出遵守答案格式的比例（`analyze_main.readout_quality`），放在論文的方法與附錄。
- **判斷題**：「誰目前比較可能說對？」，記錄 P(說出 A 的名字)。只稱為「讀出方式的分歧」，不解讀為「估計與使用的落差」。
- **推理模式（E9）**：允許思考，從最後答案解析出選擇。若可行，也讀最後答案位置的機率。

### 1.3 偏誤控制
- **孿生 episode**：名字互換、答案標籤反轉、開頭介紹名字的順序反轉；回合結果和圖片完全相同。分析時，每對的對齊 logit 取平均。
- 每回合兩位報告者的列出順序隨機。
- 情境、SV／VS 順序、開頭介紹順序三者完全交叉（以 4 個 episode 為一組）。

### 1.4 統計
- **獨立單位**是 episode 對（原版 + 孿生）。每個指標先在 episode 內平均。
- 95% CI 用 episode 層級的 bootstrap（10,000 次），檢定用 Wilcoxon。
- **等價**用 TOST，界限 ±2 個百分點。
- **情境**當作隨機因子：報告各情境的估計值，並做跨情境的符號檢定。
- dev 只用來調整設定（例如選固定視窗的 K）。held-out 只跑一次。

### 1.5 指標
| 指標 | 定義 |
|---|---|
| acc | 孿生平均後，所選報告者在該回合說對的比例（t ≥ 2）。上限約 0.8，亂猜 0.5 |
| FB | 跟隨「當下真實較可靠者」的機率，分穩定期和波動期 |
| LS | 平均 log P(真相)，proper scoring rule |
| 反轉後恢復曲線、AUC₁₀ | 只計算真正的反轉點 |
| B 的位置 | (acc_B − acc_A) / (acc_C − acc_A)。0 代表和原始紀錄（A）一樣，1 代表和摘要（C）一樣。只有 C−A > 0.05 且其 bootstrap 95% CI 下界 > 0 時才解讀（`analyze_main.position_ci`）；比值不一定落在 [0, 1] |
| 捷思吻合度 | 模型選擇和各種策略的吻合度，以扣除巧合吻合的 Cohen's κ 表示 |
| 積分核 | logistic 迴歸：過去第 k 回合「誰對」(k = 1..15)、上一回合的真相標籤（抄答案）、標籤先驗 → 模型選擇 |

---

## 2. 資料

| 資料 | 內容 | 規模 | 生成方式 |
|---|---|---|---|
| **`episodes_v3`（正式）** | 主任務。穩定期 40 回合，p_A = 0.8；波動期 40 回合，每 10 回合在 0.8／0.2 間切換，與 Behrens 2008 一致。每 10 回合固定 8 對 2，真相平衡。每個 episode 指派一個文字情境 | dev 16 + held-out 48 個 episode，加孿生共 128 個，每個 80 回合 | `generate_episodes.py`，種子 20261111 |
| `episodes_robust_v3` | 穩健性：IID 抽樣，切換間隔 6–16 回合 | held-out 24 + 孿生 | `generate_robust.py` |
| `episodes_v1` | 3090 前導資料，只用於跨平台檢查 | — | `--legacy-v1` |
| Oxford-IIIT Pet | 寵物照片（真實照片） | 384 張 | `prepare_pet_data.py` |
| Psych-101 xiong2023 | Changing bandit：30 位受試者 × 3 個 100 回合遊戲，hazard 0.1／0.2／0.4 | 9,000 回合 | `psych101/fetch_xiong.py` + `build_xiong.py` |

全部由 `portable/setup_data.sh` 重新生成，並用 `portable/checksums.json` 驗證。

### 2.1 情境（`scenarios.py`）
每個主任務 episode 跑兩個「領域槽」：
- `pet`：寵物照片，所有 episode 都跑。
- `scenario`：該 episode 指派的文字情境。

| 情境 | 報告者 | 答案 | 圖片（做法 1） |
|---|---|---|---|
| color | reporters | red／blue | 色塊 |
| weather | 氣象預報員 | rain／dry | 雲雨、太陽圖示 |
| stock | 股市分析師 | up／down | 上漲、下跌走勢箭頭 |
| sports | 球評 | win／lose | 獎盃、叉叉 |
| medical | 醫師判讀 | positive／negative | 快篩試劑的兩條線、一條線 |
| news | 消息來源 | true／false | 打勾、打叉 |
| pet | reporters | cat／dog | 真實照片 |

- held-out 中每種文字情境各 8 個 episode；dev 包含 color、weather、stock、sports。
- E1 對照會讓所有 7 種情境全部交叉。
- 圖示全部由程式畫出，不含文字。
- color 和 pet 的 prompt 和 v1／v2 逐字相同，已驗證。

### 2.2 歷史呈現方式（三步拆解）
| 代號 | 參數 | 每回合的內容 | 模型要自己做的事 |
|---|---|---|---|
| **A** | `--history-format raw` | 「Day 3: Ken forecasts rain; Sam forecasts dry. Actual weather: rain.」 | ① 判斷誰對 → ② 累加 → ③ 使用 |
| **A+** | `--history-format raw_coded` | 原始紀錄，後面再附「(Ken was correct; Sam was wrong.)」 | ②③，但保留所有原始資訊與干擾（用來區分「補上正確與否」和「刪掉干擾」） |
| **B** | `--history-format coded` | 「Day 3: Ken was correct; Sam was wrong.」 | ② 累加 → ③ 使用 |
| **B′** | `--formats chart` | 整段歷史畫成格子圖（綠色 = 說對），解析度 448 | ② 累加（用看的）→ ③ 使用 |
| **C** | `--history-mode sum_all` | 「Summary of all n past days: Ken was correct x times; Sam was correct y times.」（summarized history） | ③ 使用 |
| **D** | E1 `explicit_rel` | 「Ken was correct on 80% of past days…」（description，直接告知機率） | ③ 使用（描述） |

C 的其他變體（E3）：`sum_window:K`、`sum_cp`（BOCPD）、`sum_oracle`。

### 2.3 當下報告的格式
- `text`
- `image`：圖示，或 pet 的照片
- 綁定對照：`image_named`、`image_composite`、`image_after`

### 2.4 設計參數的依據
（內容不變，見 git 歷史中的 v2 §2.5，重點摘錄如下。）
- **80%／20%、每 10 回合切換**：等同 Behrens 2008 的波動期。規範模擬顯示，在 65%–85% 範圍內結論都相同。
- **40 + 40 回合**：兩期等長，prompt 長度在所有模型的範圍內。
- **hazard 1/15**：介於兩期長度之間；在 1/8 到 1/30 之間，準確率差異都在約 1 個百分點內。
- **episode 數**：每個配對差的 SD 約 0.05–0.065，n = 48 時以 80% 檢定力可偵測約 2–2.6 個百分點。前導觀察到的落差約 19 個百分點。
- **報告者名字**：16 個單一 token 的常見英文名。
- **照片**：Oxford-IIIT Pet，授權為 CC-BY-SA-4.0。dev 和 held-out 分別用不同的官方分割。

**規範模擬**分兩層：
1. `design_sim.py`（`runs/design_sim.json`）是**參數設計用**的抽象模擬：IID 抽樣、BOCPD 用 posterior predictive，並掃描 h 和 K 取最佳值。20 種設定中，BOCPD 減最佳固定視窗的差距都在 −0.004 到 +0.008 之間。
2. `design_sim_runner_rule.py`（`runs/design_sim_runner_rule.json`）用 **runner 實際給 VLM 的規則**計算：`history_window` 的 MAP 視窗，最少 3 回合，多數計數，平手記 0.5，h 固定 1/15。資料用真正的 v3 episode（分層）和 robust v3（IID），以 episode bootstrap 計算 CI。held-out 的結果：

| 規則 | 準確率 | 95% CI |
|---|---|---|
| sum_window:5 | .692 | [.685, .698] |
| sum_cp | .678 | [.666, .688] |
| sum_all | .632 | — |
| sum_window:10 | .612 | — |
| sum_window:20 | .579 | — |
| sum_oracle | .744 | — |

robust 資料（IID）：window:5 .675、sum_cp .663、window:10 .635、window:20 .604、sum_all .594、sum_oracle .735。前兩名（window:5 > sum_cp）和 oracle 的位置與 held-out 相同，但其餘的排序**不同**：held-out 是 sum_all（.632）> window:10（.612）> window:20（.579），robust 是 window:10（.635）> window:20（.604）> sum_all（.594）。
- 結論只限於測試過的設定：「變化點規則接近、但略低於最佳固定視窗」，不擴寫為「所有線上規則等價」。
- `sum_oracle` 的定義：只承認截至 t−1 已發生的反轉，所以是「延遲一回合得知分段」的參考值，不是提前知道當下狀態的 oracle。
- 0.8 是「已知可靠度」的參考值，不是所有可用策略的嚴格上限：固定每 10 回合 5 個 0、5 個 1 的真相配額，本身就允許「只看配額」的捷徑，dev 準確率 .661（R05），所以 robust（IID）資料的確認很重要。

---

## 3. 實驗

### E1　正向對照（dev，`*__dev_controls`）
- **條件**：7 種情境 × text／image × t ∈ {20, 40, 60, 80} × 32 個 dev episode（含孿生）。

| 對照 | 給模型的內容 | 對應的步驟 |
|---|---|---|
| perception | 單張圖：「這張圖是哪一種？」 | 看得懂 |
| single_trial_who | 只給一回合的原始紀錄，問「這回合誰說對？」 | 只測步驟 ①（單回合判斷） |
| explicit_rel | 「Ken 80%、Sam 20%」 | ③ 使用（描述） |
| explicit_counts | 「最近 10 回合 Ken 8 次、Sam 2 次」 | ③（給定的計數摘要，不是描述） |
| easy_short／easy_short_coded | 10 回合，A 全對；原始紀錄版和 ✓／✗ 版用同一組亂數 | ①②③ vs ②③ |
| easy_long／easy_long_coded | 70 回合 A、B 各半，再 10 回合 A 全對 | 長歷史下的 ①②③ vs ②③ |

- **規模**：約 11,600 題／模型（見 §5）。
- **解讀**：
  - explicit_rel（D）高、easy_long（經驗）低 → 有描述—經驗落差（D−A 類比較；只用 E1 的指標，不和主任務的 C−A 混用）。
  - coded 版回升 → 卡在步驟 ①。
  - explicit 也低 → 這個模型不會用可靠度，不進入後續的學習率分析。

### E2　主任務：經驗條件（`*__{dev,ho}_full_{text,image,chart,coded}`）
- **條件**：完整歷史，分成 A（text、image）、B（coded，text 和 image）、B′（chart）；兩個領域槽（pet、scenario）。A-text 另加判斷題。
- **測量**：acc、FB、LS、恢復曲線，以及**行為描述分析**（`analyze_mechanism.py`：捷思 κ、積分核；只寫「與某策略一致」，不推論機制）。
- **學習率分析**（`cogmodels.py`）：只對「E1 中 explicit_rel ≥ 0.8，而且 E2 有追蹤（FB 穩定期 ≥ 0.6）」的條件做。
  - 用 leave-one-episode-out 交叉驗證比較模型。
  - 有效學習率差異用 M1 虛無分佈校正。
- **圖表格式（B′）**：只對 E4b 中 chart_read ≥ 0.8 的模型解讀。

### E3　摘要條件與外部累加（`*__{dev,ho}_{sum_*, cp_hard, window10}`、`*__rob_*`）
- **條件**（全部是 summarized history 的變體，不是描述條件）：`sum_all`（**主要的 C**）、`sum_window:{5,10,20}`、`sum_cp`（加 image）、`sum_oracle`、`cp_hard`／`window:10`（截短的原始紀錄）。
- **主要檢定**：見 §0（`sum_all` − 原始紀錄）。
- **次要檢定**：
  - (a) TOST：VLM + 摘要 ≈ 規則單獨預測，對 sum_all、sum_window:K*、sum_cp 各做一次。
  - (b) 規則之間的差異，報告效果大小並和規範模擬對照，不作檢定。
  - (c) B 的位置 (B−A)/(C−A)。
  - (d) 穩健性資料上重做主要檢定。
- **K 的選擇**：在 dev 上取 acc 最高者，凍結後才跑 held-out。

### E4　多模態（dev，探索性）
- 目前只在 dev 上跑，所以結果標為**探索性**。若要在正文支持「綁定失敗」，需要加跑凍結後的確認資料。
- 措辭要和已排除的解釋一致：composite_read 讀對，只排除了「單張合成圖的基本辨識」問題；chart_read 只檢查單格辨識，不代表模型能跨整張圖累加（R09）。
- **E4a 照片綁定**（`*__dev_binding`）：explicit_rel × {image_named, image_composite, image_after} × 所有情境，加上 composite_read。依 A 排第一或第二分開報告，並且分模型報告。
- **E4b 圖表可讀性**（`*__dev_chartread`）：「第 k 回合誰說對？」。chart_read ≥ 0.8 是解讀 B′ 結果的前提。

### E5　跨平台檢查（`*__xcheck`）
- MI300X 和 3090 的同一批 prompt 比較：|Δy| 中位數 < 0.02，且 argmax 一致率 > 98%。
- 既有 xcheck 是 Qwen 8B／32B 的 v1 文字子集；通過只驗證該範圍。3090 遷移另需檢查有代表性的圖片與長歷史 prompt，不能把文字子集的結果當成所有圖片／模型已驗證。任何新增驗證另行記錄，不改原 E5 門檻。

### E6　Psych-101 外部任務（`*__psy_*`，細節見 `psych101/PROTOCOL.md`）
- **資訊和經驗不對等（H01）**：人類被告知 h，而且之前有 1,600／3,200 回合的經驗；模型的主要條件是 hidden。所以「hidden 模型 vs told 人類」是**描述性的跨條件比較**，不能單獨證明人機學習能力的差距。另外報告 told 模型 vs told 人類，以及模型自身的 told − hidden。人類的 learning-rate 結果是**本專案的重新分析**，不是原論文直接報告的結果。
- **校正（H02）與決定**：用模擬資料檢驗了完整流程（人類擬合、20 次抽樣擬合、配對斜率檢定；`psych101/calibration.json`，200 seeds × 3 arms）。結果：δ = 0 時偽陽性率 6.0% ± 1.7%（MC SE），95% CI 涵蓋率 0.935；20 次抽樣的中位數估計量有約 +0.09 的向上偏差；|δ| = 0.5 時檢定力只有 0.49–0.59（δ = −0.5 時涵蓋率降到 0.87）。
  - **決定**：人機比較（C4）降為**輔助性、描述性**的結果。報告配對斜率差的估計值與 bootstrap 95% CI，並一併報告上述校正結果（偽陽性率、偏差、涵蓋率、檢定力）。
  - **不做**人機「有差異」或「相似」的強主張。差距在 0.09 左右時可能完全來自估計偏差；在這個檢定力下，不顯著的結果無法解讀（不能當作「人機相同」的證據）。
  - 偏差校正或 parametric-bootstrap null（PROTOCOL.md 的 proposed fix）若在凍結前實作，只作為敏感度分析，不改變上述主張層級。
- **方法**：重播人類自己的歷史；**hidden**（不告知 hazard）是主要條件；格式為 text、image_outcome、chart，另有知覺對照。
- **模型 φ 的擬合**：從模型機率抽樣出選擇序列（每個 episode 20 條），用和人類相同的模型擬合。
- **E6 內的主要比較**（對整篇論文而言是輔助性的）：每位受試者的 φ 對 h 斜率，模型 vs 人類（同一份歷史，配對 n = 30），報告中位數差與 bootstrap CI，以及 Wilcoxon p；解讀受上面 H02 的決定限制。
- **人類基準**：100 回合遊戲的斜率 Wilcoxon p = 0.0076（Page p = 0.017）；1500 回合遊戲 p = 0.029，控制順序後的 CI 跨過 0。兩者都會報告。
- **限制**：模型看到的是獨立的「Game 1」，人類在那之前已有經驗。

### E7　API 模型（主任務定案後才跑）
- 只跑 E1 對照，或其他已確認有價值的子集。讀出方式不同（logprob 或多次取樣），論文中分開呈現。

### E8　規範模擬（不用 GPU）：見 §2.4。

### E9　推理模式（`run_thinking.py`；工作見 `portable/make_jobs.py` 的 `*__e9_*`）
- **模型與模式**（與 jobs 一致）：
  - `qwen3vl8b_thinking`、`qwen3vl32b_thinking`：`--mode thinking`（原生思考模板），作為補充。
  - `qwen3vl32b`：`--mode cot`（同一 checkpoint 加 CoT 提示）。這是**主要比較**：和同一 checkpoint 的直接作答（E1／E2 的 qwen3vl32b）配對，因為 Thinking 和 Instruct 是不同的 checkpoint。
- **資料**：dev split（`episodes_v3`，`--split development`），`--domain scenario`（每個 episode 用自己指派的文字情境；不跑 pet），只用 text 格式。
- **子集**：
  - E1 對照：`explicit_rel`、`easy_short`、`easy_short_coded`、`easy_long`、`easy_long_coded`，查詢回合 t ∈ {40, 80}（`--control-t 40 80`）。種子與 run_controls 相同，所以和直接作答的對照題逐字相同。對照只跟 raw 工作一起跑一次。
  - 主任務：回合 t ∈ {20, 40, 60, 80}（`--main-trials 20 40 60 80`），歷史分兩種：A（`--history-mode full --history-format raw`）和 C（`--history-mode sum_all`）。
- **生成設定**：`--max-new-tokens 16384`，每個模型 × 歷史 4 個 shard（`--shard i/4`），共 3 模型 × 2 歷史 × 4 shard = 24 個工作（tier 2）。
- **分析**：`analyze_thinking.py` 已實作並納入 `prereg-v3`。E9 結果仍須通過列數、有效答案與截斷等完整性檢查後才報告；分析入口存在不代表 E9 模型工作已完成。
- **測量**：準確率、推理長度、是否被截斷、C−A 落差和 D 對經驗的差距是否縮小。
- **解讀**：結論分成「單次作答」和「逐步推理」兩部分報告。
- **測量細節（R07）**：
  - 分開報告以下幾項：自由生成答案的準確率、有效答案率（分母包含解析失敗）、被截斷的比例、強制補答的比例、token 成本。
  - 被強制補答的列不算自然答案。推理後讀到的 P 是「在已生成的推理之後」的條件機率，和直接作答的 P 不是同一種量；也要報兩個候選 token 的機率質量。
  - 若 CoT 改變了核心結論，必須寫進正文，不能只放附錄。

---

## 4. 主要圖表（8 頁版；與 `paper/main.tex` 一致）
1. **圖 1**：任務示意（多情境；歷史格式 A／A+／B／B′／C 與描述條件 D，以及各自留給模型的步驟）。
2. **圖 2**：主任務準確率，依歷史格式（A、A+、B、B′、C）分模型，附各情境的散點和規則單獨預測的參考線。D（直接告知可靠度）放在**另一個 panel**，用 E1 自己的指標，不和主任務 acc 共用縱軸。
3. **圖 3**：恢復曲線（A、C 的各種變體、oracle）。
4. **圖 4**：行為描述。積分核和捷思 κ（A vs C）。
5. **圖 5**：Psych-101，φ 對 hazard，人類（told）vs 模型（hidden；told 為次要）；輔助性、描述性。
6. **表 1**：模型。**表 2**：主要和次要檢定（CI、p、TOST）。
7. **附錄**：E1 對照的完整熱圖（模型 × 對照 × 情境）、綁定（E4）、推理（E9）、規範模擬（E8）、穩健性、跨平台檢查、學習率擬合、讀出品質（候選 token 質量、格式遵從率）、3B／4B 前導結果、各情境的明細。

## 5. 執行分級與計算量
- 啟動器用 `--tier` 依序執行：
  - **tier 1** 是核心證據鏈：E1、綁定、chart_read、跨平台檢查、A／A+／B／C 的 dev 和 held-out、robust 的 full 和 sum_all。
  - **tier 2** 是解釋性條件：image、chart、sum_cp、sum_window、oracle、Psych-101 文字版、E9。
  - **tier 3** 是探索性條件：cp_hard、window10、Psych-101 的圖片和知覺對照、72B。
- 原 MI300X 清單的主任務題數：tier 1 約 0.71M，tier 2 約 0.76M，tier 3 約 0.73M（全部模型合計，不含對照、Psych-101 和 E9）。E1 每個模型約 11,600 題。3090 預設排除 72B，實際工作數以 `jobs_3090.jsonl` 為準，不沿用這組總量估算。
- 先跑 tier 1 的 dev 和 xcheck，確認核心鏈可用後，再跑 held-out（W02）。

### 原估計（保留供參考）
| 項目 | 題數 |
|---|---|
| E1 | 約 7,500（已過時；實際約 11,600，見上） |
| E4 | 約 3,500 |
| E2 | dev 約 31,000、held-out 約 92,000 |
| E3 | 約 250,000（摘要類 prompt 較短） |
| E6 | 約 37,000 |
| E9 | 另計（推理很長，只跑子集） |

## 6. 前導結果（3090，v1，n = 16，僅供參考）
- 32B 的原始紀錄—摘要落差（不是描述—經驗落差；前導沒有在主任務上跑 D）：sum_cp 相對原始紀錄 +0.191，95% CI [0.161, 0.220]；sum_all 相對原始紀錄 +0.145。
- TOST：VLM + 摘要和規則單獨預測等價，差 +0.001。
- 原始紀錄下，與「抄上一回合結果」捷思的吻合度 κ = 0.25（32B），積分核幾乎是平的；用摘要時呈現指數衰減（只是 v1 前導的描述，不是機制）。
- 照片綁定：32B 照片 P = 1.00（A 排第一）對 0.03（排第二）；InternVL-8B 在照片上約 0.80。
- 圖表可讀性：Qwen3-VL-8B 的 chart_read = 0.56。

## 7. 決策紀錄
**凍結**：tag `prereg-v3` 的內容與凍結前已看過的東西，見本文件開頭的「凍結程序」。計畫的定位是「在 held-out 評估之前固定」（fixed before held-out evaluation），**不是** preregistered。以下每條紀錄都標明是否在看到 held-out 模型結果之前。

- 2026-11-11：從 source_trust 轉到 volatile_trust；目標改為 8 頁 archival。
- 2026-11-12：
  - 程式碼審查修正，產生 v2。
  - 規範模擬顯示不需重新設計任務。
  - 模擬審稿和新意檢查之後改為 v3：主軸改為三步拆解（當時稱「描述—經驗落差」；依下一條的術語調查更正為：主要檢定是原始紀錄—摘要落差 C−A，描述—經驗落差只指 D−A），並加入多情境、coded 格式、Gemma、推理模式、機制分析、Psych-101 以 hidden 為主要條件。
  - 依術語調查：C 改稱 summarized history，D（explicit_rel）才是 description。必須引用 Krishnamurthy et al. 2024（bandit 中的 raw vs. summarized history）並說明差異。Behrens 波動期的描述以 2008 年的 Supplementary Information 為出處，寫成「adapted from」。
  - 主要 C 選定為 `sum_all`。理由：它由同一段歷史計算，不引入額外觀察，也不推斷變化點。（原寫「資訊量相同」有誤：計數丟掉時間順序，不是無損，見 §0。）此決定在看到任何 v3 主任務結果之前做出。
- 2026-11-12（Codex 外部審查之後；**尚未看到任何 held-out 模型結果**；v3 上看過的只有 Qwen3-VL-8B 的部分 dev E1 對照結果，見本文件開頭的揭露）：
  - 修正執行面：I01–I07，包括保存原始 logit、續跑截斷、指紋檢查、嚴格的完成判定、答案解析。
  - 修正分析面：A01–A05，包括完整性檢查、指紋分組、TOST 的 Holm 校正、跨情境只用文字情境、B 位置的穩定性。
  - 修正措辭：C 不是「無損」；C1 拆成三個部分；C2 區分三種等價；E4 標為探索性；Psych-101 是描述性的跨條件比較。
  - 新增 A+（raw_coded）和 single_trial_who。
  - 新增執行分級。
  - 規範模擬改用 runner 實際規則重算。
  - **揭露**：規則基準（不含任何模型）在 held-out 資料上計算過，見 §2.4。這只是資料生成器的性質，不涉及模型行為。
  - **揭露**：真實模型在少數 held-out 題目上跑過 smoke test，只為了檢查管線，輸出沒有分析。
- 2026-11-12：使用者決定不做正式的預先登錄；論文以「分析在 held-out 之前固定」的方式描述。
- 2026-11-12（稽核 `discussion/codex_audit.md` 之後；**尚未看到任何 held-out 模型結果**）：
  - 術語：清除剩下把 C 稱為「描述」的地方（B 位置、前導、決策紀錄、E1、圖表計畫）；C1 只指 C−A，D−A 另外報告。
  - 預寫結論改成中性描述（C1c、C3、論文標題與圖說）。
  - H02 決定：人機比較降為輔助、描述性（E6）。
  - E9 規格改成與 jobs 一致；當時分析入口 `analyze_thinking.py` 尚在撰寫，後已完成並納入 `prereg-v3`。
  - 圖表計畫與 main.tex 統一為 5 圖 2 表；E1 熱圖移到附錄。
  - 凍結程序列出 tag 內容與凍結前已看過的東西（本文件開頭）。
- **原備案**：如果 Gemma 授權未通過，就用 Mistral-Small-3.2-24B 替代作為獨立家族。2026-11-12 已通過 Gemma 4B／27B 存取授權並下載固定版本權重，未啟用 Mistral 替代。
- **偏離 D1（2026-11-12，凍結 tag `prereg-v3` 之後；依據只有 dev 資料，尚未看過任何 held-out 模型結果）：medical 和 sports 的 icon 換新（`ICON_VERSION = 'v3.1'`）。**
  - **原因**：AMD dev 控制組的單圖辨識（perception）中，三個模型認 medical icon（試紙）都只有 .50；InternVL3.5-8B 認 sports icon（獎盃對打叉的灰球）也只有 .50。也就是模型認不出圖，圖片報告條件的結果會混入「看不懂 icon」，無法解讀成綁定或學習的問題。
  - **怎麼選**：只用 dev 控制題（`icon_probe.py`、`icon_probe_controls.py`；Qwen3-VL-8B 與 InternVL3.5-8B）。規則是先要求兩個模型的單圖辨識都 = 1.00，再挑圖片控制題（明示可靠度、短 raw、短 coded、長 coded）平均最高的。medical 試了 5 種、sports 試了 4 種。結果：medical 用加號／減號（環形外框、同色），sports 用笑臉／哭臉。彩色加減號的下游平均較高（.738 對 .668），但 InternVL 的辨識只有 .50，所以不選。
  - **影響範圍**：只有顯示 icon 報告的工作（`dev_controls`、`dev_binding`、`{dev,ho}_full_image`、`{dev,ho}_full_coded`、`{dev,ho}_sum_cp`）。這些工作用 `portable/supersede_icon_v3.py --apply` 移到 `runs_mi300x/_superseded_icons_v3/` 後重跑。文字報告、pet 照片、圖表、Psych-101、robust、E9、xcheck 不受影響，不必重跑。主檢定（文字報告 C−A）完全不受影響。
  - **防呆**：`run_volatile.py` 遇到含圖片格式、但 icon 版本不同的既有輸出時，拒絕續跑；`run_controls.py` 的 fingerprint 加入 `icon_version`。舊版 icon 用 `icon(..., version='v3')` 可以重現。
  - 論文附錄 G 報告所有候選 icon 與結果（含原版），舊 icon 的 dev 結果保留在 superseded 目錄，不刪除。
- **探索性附加：提示措辭穩健性（2026-11-12，凍結之後；只用 dev，不屬於任何預定檢定）。** `paraphrase_probe.py` 不改 runner，替換提示句的文字；兩個 8B 模型 × 四種文字歷史格式 × 三種措辭（orig 本機重跑、p1 換句話說、p2 精簡表格 `#t | Ken=dog | Sam=cat | truth=dog`）。結果（`analyze_paraphrase.py` → `analysis/paraphrase/summary.txt`）：orig 與 MI300X 差 ≤ .003；p1 全部結論保留（摘要−原始 +.12／+.15）；p2 原始歷史升到 .55–.56、逐回合標正確與摘要降到 .56–.60，InternVL 跟摘要規則一致率 .72（原本 .98），差距縮成 +.03 [.01,.06]／+.01 [−.02,.05]。論文附錄 J（tab:wording）和 Limitations 照實寫：差距大小依歷史排版而定，主結論只針對自然語言排版。不影響 AMD 任何工作。
- **措辭拆解與擴充（2026-11-12，探索性，只用 dev）。** p2 拆成三個組件：H（任務說明）、L（歷史每行格式與摘要句）、Q（當前回合行與提問），做 2×2×2 全因子實驗（`paraphrase_probe.py --wording H<x>-L<x>-Q<x>`）。已確認 Ho-Lo-Qo 的 prompt hash 和原 runner 逐字相同。分層設計：每個家族較小的模型做完整拆解，包括 Qwen3-VL-8B、InternVL3.5-8B（3090，原始歷史、逐回合標正確、摘要三種格式）和 Gemma-3-12B（MI300X）；較大的 Qwen3-VL-32B、InternVL3.5-38B、Gemma-3-27B、Qwen2.5-VL-72B 只跑 p1／p2 × 四種歷史，原始措辭的基準用各自的 dev_* 工作。MI300X 上的工作列為 tier 4，放在 `jobs.jsonl` 最後（`make_jobs.py`），README §5c。分析用 `analyze_paraphrase.py`，會輸出各組件的主效果。
- **遮罩實驗（2026-11-12，探索性，只用 dev，3090）。** 問題：逐回合標正確的歷史（raw+correct）只收回原始到 coded 差距的 1/5–2/5，而且照抄上一回合結果的權重反而上升。是結果標籤、報告標籤，還是長度造成的？`masking_probe.py` 不改 runner，沿用 raw+correct 每行的排版和長度，把部分欄位換成 `---`：rep_ok（遮結果）、out_ok（遮報告）、mask_ok（兩者都遮，資訊等同 coded，但長度接近 raw+correct）。兩個 8B 模型 × 3 種，對照組用同機的 `runs/paraphrase/orig_*`。分析用 `analyze_masking.py`（2×2 主效果與交互作用、mask_ok−coded 檢查長度／排版、copy 權重與 κ）。
  - 結果（兩個 8B）：遮結果的主效果 +.07／+.10，遮報告約 0，交互作用 +.07／+.05；兩者都遮跟 coded 只差 −.02 → B−A+ 差距約 80–85% 來自歷史中的標籤，長度／排版約 15–20%。copy 權重只要顯示結果就維持 .27–.50，遮掉後 ≤ .12。詳見 `discussion/masking_analysis.md`。MI300X tier 4 加入 15 個 `*__mask_*` 工作（Gemma-3-12B 與四個大模型），`jobs.jsonl` 前 328 行不變。
- **偏離 D2（2026-11-12，凍結 tag `prereg-v3` 之後）：先準備 RTX 3090 主力執行版本。**
  - **原因與範圍**：使用者指出 MI300X 須優先供其他 CVPR 工作使用，H200 需要排隊，因此授權先完成 3090 配置。這是算力可用性決定；不以哪個平台結果較佳作選擇依據。本次只準備清單與檢查，未據此啟動新的模型工作。
  - **模型與卡數**：核心五個模型保留，包含獨立 Gemma 家族的 Gemma-3-27B。兩個 8B 各配一張 3090；Qwen 32B 配四張，已有本機 BF16 執行紀錄；InternVL 38B／Gemma 27B 暫配四張，仍需實測顯存。Gemma 12B、Thinking 8B 暫配兩張，Thinking 32B 暫配四張；E9 須另測長生成峰值。72B 為選配延伸，不進入 3090 預設清單，可待 H200 等算力可用後另行安排。
  - **保存與可重現性**：由 `make_jobs.py --platform rtx3090` 產生 `jobs_3090.jsonl`，輸出 `runs_3090/`；原 `jobs.jsonl` 和 `runs_mi300x/` 保留。模型 revision、BF16／FP32 讀出設定、資料、icon v3.1 和分析方法保持一致。同模型各正式條件使用同一平台，不以 AMD 結果接續 CUDA 輸出；跨平台比較另依 E5 及新增圖片／長歷史檢查報告。
  - **目前完成程度**：本機正式資料已備齊；Gemma 27B 固定版本權重已下載完整，核心五模型共 243 個工作的資料與權重 preflight 通過。選配 Gemma 12B、Thinking 32B／72B 仍未備齊。唯讀 preflight、指令 dry run、真正的模型載入／推論驗證三者分開記錄，前兩者不證明大模型顯存足夠。
  - **結果揭露**：本次配置工作沒有檢視 held-out 模型輸出；其他既有工作的執行與分析狀態仍須依紀錄核對，因此不新增「此決定早於所有 held-out 模型結果」的主張。既有凍結時間與 D1 揭露保留，D2 不回溯改寫。
- **探索性附加（2026-11-12，3090 小模型 v3 dev 計畫）**：依使用者要求，另準備 Qwen3-VL-4B、InternVL3.5-4B、Gemma-3-4B 與兩個 8B 的前導清單 `jobs_3090_small_dev.jsonl`，共 80 個工作。以 `make_jobs.py` 的 `--platform rtx3090`、明確指定五模型的 `--models` 和 `--dev-only` 生成，只含 `__dev_*` 工作，`--out` 指定獨立清單（完整指令見 README §5）。Gemma 4B 固定 revision `093f9f388b31de276ce2de164bdc2081324b9767`，僅作獨立家族的探索性 dev 比較，不取代正式核心 Gemma 27B。Qwen／InternVL 4B 快取與 processor 離線檢查可用；Gemma 4B 固定版本的兩個權重分片已完整下載、preflight 通過，CPU 檢查通過 56 個單 token 候選答案及兩張圖的 `2 × 3 × 896 × 896` 輸入，pan-and-scan 關閉。v3 GPU 推論、顯存峰值及結果仍待驗證。相同 8B 工作沿用 `runs_3090/` 目錄，不重複建立同條件結果。這項準備不改預設 292 個工作、核心五模型、獨立 Gemma 家族要求或正式 held-out 檢定，也不預先宣稱尺度效應；既有 3B／4B v1 前導資料仍依 §1.1／§6 的限制呈現。1B／2B 尚未加入。

- **執行與分析完整性修正 D3（2026-11-12，凍結 tag 之後的流程稽核）**：本次只檢查程式、資料、CPU 合成測試及既有 dev 輸出的相容性，未檢視 held-out 模型輸出；不回溯變更原凍結內容。
  - **啟動／續跑**：不能僅憑任意 `_COMPLETE*` 檔跳過工作，須核對正確 shard 的完成證據、列數／唯一性、模型 revision、參數及資料 fingerprint。新增 launcher 配置紀錄及可由子程序繼承的原子檔案鎖；任一工作失敗、完成證據無效或仍有 writer 時整輪回傳非零。舊 icon 遷移工具兼容新／舊鎖格式且拒絕覆寫既有封存。
  - **dev 與正式資料的界線**：三個探索性 4B 模型必須以 `--dev-only` 產生清單。固定 K 的輸出／載入須使用完整 dev 的 K=5、10、20、兩個領域槽及一致的模型版本／讀出，拒絕 partial 或缺少候選。操作順序改為先完成 dev、保存 K，再啟動 held-out。
  - **檢定家族**：修正同時輸入主資料與 robust 時，把 robust 誤納入主要 Holm 家族的程式行為。robust 依 E3(d) 維持獨立的次要穩健性檢定與 Holm 家族；主資料的主要檢定不因此增加成員，效果定義與統計方法不變。
  - **Psych-101**：正式分析在擬合前要求每個 choice 條件的 30 人 × 3 hazards × 100 回合及完整知覺控制，並核對資料 hash、fingerprint、revision 和結果列。明確的 `--partial` 僅用於探索，不能標為主要結果。
  - **驗證範圍**：修正後完整 CPU 回歸 169／169 通過（107.24 秒），Psych-101 runner 的獨立 CPU 續跑自測亦通過。三份清單 343／292／80 個工作的真實 CLI 參數解析全部通過；原 MI300X 與 3090 清單內容未因 D3 改變。核心五模型 243 個工作與小模型 80 個工作的資料／權重 preflight 通過。八個已下載 Instruct 模型均通過 56 個單 token 候選與第 80 回合照片／medical／sports 圖片的離線 processor 檢查。這些均非 GPU 推論結果；本次七張 3090 都已有程序，未啟動新的 GPU 工作。新模型載入、多卡顯存、E9 長生成及圖片／長歷史跨平台比較仍須實測。
- **3090 dev 開始排程（2026-11-13 00:07，Asia/Taipei）**：依使用者「把可以排程下去跑的部分先拿下去跑」的授權，啟動本機背景 dispatcher（初始 PID `3543087`）。固定批次 `runs_3090/_queue/dev_20261112/` 合併核心五模型與三個探索性 4B，共 128 個唯一 dev 工作、每模型 16 個；兩個 8B 不重複跑。沿用既有提示、icon v3.1、固定 revision 與 BF16／FP32 讀出，不新增 1B／2B，不在此批次執行 held-out／robust／Psych-101／E9 或新措辭搜尋。
  - **派工與隔離**：本機沒有 Slurm 指令，改以 detached queue 每 30 秒掃描 GPU，連續空閒 60 秒才派工；最多同時兩個工作，四卡模型須取得四張空卡。GPU 與輸出鎖繼承至 runner；每模型先做 26 題 dev 技術 smoke（長歷史、兩圖、chart、綁定等），驗證數值、精度、無 offload 與各卡至少 512 MiB 顯存餘裕，通過後才跑該模型 dev。smoke 不以準確率篩選模型，輸出與正式結果分開；失敗停止該模型並保留診斷。
  - **驗證與初始狀態**：128 個工作的資料／權重 preflight 通過；排程、launcher、smoke 等 82 項 CPU 測試全過，Qwen／InternVL／Gemma 原生 processor 的 26 案例檢查通過。啟動時七張卡都忙碌，因此狀態為 `waiting`（完成 0／128、GPU 子程序 0），尚未產生本批次 GPU 結果。後續以 `status.json` 與 `queue.log` 為準；來源 hash、程式／資料快照、啟動參數和環境版本均保存於該批次，監看來源變動即停止新派工。dev 完成後仍須驗收並固定 K，才安排 held-out。

- **縮小目前 3090 排程（2026-11-13 00:14，Asia/Taipei）**：依使用者最新指示「27 B以上的就先不要排了」，暫緩 Gemma 27B、Qwen 32B、InternVL 38B 的 48 個 dev 工作。原 dispatcher 已在沒有任何模型工作啟動的情況下正常停止；原批次與凍結清單保留，不覆寫既有執行紀錄。新批次 `runs_3090/_queue/dev_4b8b_20261113/` 僅含 Qwen／InternVL 的 4B、8B 與 Gemma 4B，共 80 個唯一 dev 工作（各 16 個，tiers 35／30／15），完整設定與原清單對應工作逐項相同，資料／權重 preflight 80／80 通過。新背景 dispatcher PID `3564099` 已啟動，初始為 `waiting`、完成 0／80、執行中 0；沿用原技術 smoke 與空卡檢查。此次是執行範圍調整，不移除研究計畫中的大模型或下載權重，也不產生新的結果主張；27B 以上待後續指示再排。

- **依空卡數優先派工與可續跑交接 D4（2026-11-13，Asia/Taipei）**：使用者重新授權四卡時優先 Gemma 27B／Qwen 32B／InternVL 38B，兩卡時優先既有計畫內最大的可容納 BF16 模型；後續明確要求將本方低卡數工作釋放後直接交接四卡。原 80-job 批次在沒有模型工作執行中的情況下停止。新 `runs_3090/_queue/dev_adaptive_20261113/` 清單共 144 個唯一 dev 工作（九模型各 16 個，tiers 63／54／27）；原 128 個工作的參數與輸出不變，新增的 Gemma 12B 16 個條件為探索性 dev，不擴大正式 held-out 檢定。預設 MI300X 343／3090 292 工作清單保持原樣。
  - **兩卡模型**：Gemma 12B 固定 revision `96b6f1eccf38110c56df3a15bffe176da04bfd80` 的五個權重分片已完整下載，共 22.701 GiB；維持 BF16 主體、原生 896px processor 與獨立 FP32 輸出頭。九模型 144 個工作的資料／權重 preflight 通過，Gemma 12B 的 56 個候選單 token、26 種 dev 輸入的原生 processor CPU 檢查通過。這些不是 GPU 執行或峰值顯存證據，首次仍由獨立 smoke 放行。
  - **交接規則**：需明確啟用 `--preempt-to-four`。新空卡先通過連續 60 秒空閒與實際 GPU lease 檢查；只對本 dispatcher 啟動且 UID／session／process group 經核對的低卡數工作送停止訊號。原組退出後保留並移交 GPU lease，四卡的新鮮程序／顯存檢查通過即啟動目標，不對本方剛釋放的卡再次等 60 秒；交接期間每秒檢查，較小工作不得插入。已寫結果保留，受中斷工作返回 pending 而非模型失敗，之後依保存的題目 key 續跑。未寫入列可能重算，模型重載仍需時間；外部使用者的程序不在停止範圍。
  - **中斷安全**：fingerprint、完成標記和 launcher contract 改為暫存檔＋原子 replace；metadata JSONL 續寫前截掉不完整尾行；結果尾行採原檔 truncate，避免重寫已完成 prefix。提示、評分與結果欄位不變。非排程部分 210 項 CPU 回歸通過，包含真 SIGTERM 與兩個 runner 的中斷續跑；GPU 切換仍須等有空卡才能觀察，排程測試與實際啟動狀態另記在新批次驗證紀錄。
  - **實際啟動**：2026-11-13 02:15（Asia/Taipei）已啟動新版 detached dispatcher，PID `3903976`，明確帶入 `--preempt-to-four --preempt-grace-seconds 30`。最終 210 項非排程回歸＋40 項 queue 測試共 250／250 通過；含真 CPU process group、TERM→KILL、leader 先退出、lease 競爭、STOP／來源變更、完成競態及四卡直接交接。初始狀態 `waiting`、完成 0／144、執行中 0，七張 GPU 都已有其他工作，尚未做真 GPU 搶占／交接；17 個執行來源檔的 hash、程式與測試快照、環境版本、驗證紀錄及實際 argv 均保存於新批次。


- **H200／Slurm dev 版本準備（2026-11-13，Asia/Taipei）**：依使用者要求新增 `portable/h200/`，不修改正在運作的 3090 dispatcher 或其 17 個凍結執行來源。預設選 Gemma 27B、Qwen 32B、InternVL 38B 共 48 個 dev 工作；另提供九模型 144 個 dev 的選項。每模型一個 Slurm array task、一張完整 H200，預設最多兩個 task 並行；此為算力配置，提示、資料、icon v3.1、固定 revision、BF16 主體／FP32 讀出與科學參數沿用原清單，輸出獨立於 `runs_h200/`，不擴大到 held-out／robust／Psych-101／E9 或新措辭搜尋。
  - **執行規則**：由 Slurm 配置 GPU，保留其 CVD 與 cgroup 映射；每次 allocation 先檢查單張 H200 至少 120 GiB，再做獨立 26 題技術 smoke，通過後執行該模型 16 個 dev 條件。來源／清單 hash、模型／輸出 contract、持久 flock 與完整性檢查保護續跑。時限預警經 batch→srun→runtime 傳入，停止本方子程序群組並以 75 標示可續跑；不自行無限重送，超時後可另建計畫重送同模型，保留已完成題目。
  - **本機驗證**：H200 新增 CPU 測試 66／66 通過，包含真程序訊號、CVD／lock FD 繼承、來源變更、smoke 失敗、中斷續跑、重複提交防護與路徑隔離；Bash 語法檢查通過。九模型 144 個工作的資料與固定權重 preflight 通過，48／144 工作的 Slurm dry run 已保存於 `runs_h200/_slurm/local_review_{large,all}_20261113/`。程式與凍結輸入套件含 805 個檔案、768 張圖片，約 117.9 MiB，不含權重、憑證或結果；在乾淨臨時目錄解壓後，805 個 hash、66 個 CPU 測試、runtime preflight 與提交 dry run 全部通過，驗證紀錄在 `runs_h200/_slurm/export/`。
  - **目前界線**：尚未取得 H200 站台登入／partition／account／QoS／共享路徑，未實際提交 Slurm，也未進行 H200 模型推論或測量峰值顯存／速度。權重目前在本機，需搬到 H200 可存取的位置或重新下載。不能據此宣稱 H200 比 3090／MI300X 更快或結果等價；跨平台結果仍須獨立驗證。
- **偏離 D5（2026-11-13；遠端紀錄原標為 D2，與本機的 D2 編號重複，合併時改為 D5；凍結 tag `prereg-v3` 之後；決定時尚未看過任何 held-out 模型結果）：評估平台改成 H200、拿掉 72B、精簡工作清單。**
  - **原因**：計算資源。MI300X 不再使用；原清單在截稿前跑不完。
  - **平台**：held-out、穩健性資料、Psych-101、論文會報告的 dev 對照（E1、綁定、圖表可讀性）和選 K 用的 dev 視窗，全部在 NVIDIA H200 上跑（PyTorch 2.8.0、CUDA 12.8、Transformers 4.57.1），同一個模型的這些條件都在 H200 完成。探索性的 dev 檢查（icon、措辭、遮罩）維持在 RTX 3090。論文不報告 MI300X 的結果。
  - **一致性檢查**：每個模型在 H200 重跑 `dev_full_text` 與 `xcheck`，和既有輸出逐題比對（E5 門檻：|Δy| 中位數 < 0.02、選擇一致率 > 98%、提示雜湊相同），只作內部檢查，不放進論文。`portable/h200/xplat_check.py`。
  - **模型**：拿掉 Qwen2.5-VL-72B（規模延伸）、Qwen3-VL-32B-Thinking 與 32B 的 E9 CoT、Gemma-3-12B；E9 只留 Qwen3-VL-8B-Thinking。
  - **工作**：依論文主張重新分級，共 177 個（`portable/h200/PLAN_H200.md`、`jobs_h200.jsonl`）；拿掉大模型的措辭 p1、穩健性資料的額外條件（`rob_full_coded`、`rob_sum_cp`、`rob_sum_window10`）。主要檢定（C1，held-out C−A）與其分析方式不變。
  - **執行**：`portable/h200/worker.py`（依清單順序跨模型挑工作）。
- **取消 RTX 3090 執行計畫（2026-11-13，使用者決定；尚未看過任何 held-out 模型結果）**：偏離 D2（3090 版）、3090 小模型 dev 計畫、D4 與相關排程全部取消；本機 dispatcher 已以 `STOP` 停止，跑到一半的工作已終止。`runs_3090/` 的輸出不用於論文。3090 專用的程式修改（`make_jobs.py`、`models.json`、`run_jobs.py`、`README_RUN.md`、`supersede_icon_v3.py`、`run_controls.py`、`run_volatile.py`）還原成 commit 版本，未提交的原始內容保存在本機分支 `backup/3090-local-20261113`；runner 不變，H200 續跑不受影響。D3 中與平台無關的分析完整性修正保留並提交：`analyze_main.py`（穩健性資料的檢定自成次要 Holm 家族、K* 只能由完整 dev 的 K=5／10／20 選出並核對）、`psych101/analyze_xiong.py`（正式分析前檢查完整設計與 fingerprint），`test_analysis.py` 49／49 通過。論文中先前在 3090 上做的探索性 dev 檢查（icon 選擇、措辭、遮罩）照原樣保留。
