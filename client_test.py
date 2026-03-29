import requests
import pandas as pd


def call_nllb_api(payload):
    url = "http://127.0.0.1:8787/evaluate"

    print(f"正在請求評估: {payload['corpus']} ... (這可能需要一點時間)")

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()

        if result["status"] == "success":
            data = result["data"]
            print("\n✅ 評估完成！")
            print(f"指標類型: {data['metric']}")
            print("-" * 30)

            # 1. 顯示 Summary
            print("總結分數 (Summary):")
            for k, v in data["summary"].items():
                print(f"  {k}: {v}")

            # 2. 將 Details 轉成 DataFrame
            df = pd.DataFrame(data["details"])
            print("\n詳細語言分數 (Details Table):")
            print(df.to_string(index=False))

            # 3. 儲存成 CSV (選配)
            # csv_name = f"result_{payload['pivot']}_{payload['corpus']}.csv"
            # df.to_csv(csv_name, index=False)
            # print(f"\n💾 結果已存檔至: {csv_name}")
            return df

        else:
            print("❌ Server 回傳錯誤:")
            print(result.get("stderr"))
            return None

    except Exception as e:
        print(f"⚠️ 連線 API 失敗: {e}")
        return None


if __name__ == "__main__":
    # 你要傳入的參數
    payload = {
        "corpus": "flores200_7",
        # "translate_dir": "/alghome/timmy.wan/translation/nllb-3.3b-jaenmsthvitlid-translate",
        "translate_dir": "/alghome/timmy.wan/translation/vllm-lmt-0.6b-jaenmsthvitlid-translate",
        "pivot": "jpn_Jpan"
    }
    res = call_nllb_api(payload)
    if res is not None:
        lang_to_xx = res.iloc[:, 1].mean()
        xx_to_lang = res.iloc[:, 2].mean()
        print(f"{payload['pivot']}-xx: {lang_to_xx:.6f}")
        print(f"xx-{payload['pivot']}: {xx_to_lang:.6f}")
        print(res)
        import pdb; pdb.set_trace()
        print("-------------------")

"""
(qwenomni) timmy.wan@Alg5:~/translation/Open-NLLB$ python api_server.py
(base) timmy.wan@Alg5:~/translation/Open-NLLB$ python client_test.py

正在請求評估: flores200_7 ... (這可能需要一點時間)

✅ 評估完成！
指標類型: spbleu_flores200
------------------------------
總結分數 (Summary):
  jpn_Jpan-xx: 8.983333
  xx-jpn_Jpan: 4.416667

詳細語言分數 (Details Table):
tgt_lang  jpn_Jpan-xx  xx-jpn_Jpan
tgl_Latn          2.4          2.1
zsm_Latn          7.4          3.7
ind_Latn          8.9          4.2
vie_Latn         11.3          4.3
tha_Thai          5.6          2.6
eng_Latn         18.3          9.6
"""
