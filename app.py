import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import google.generativeai as genai

# ==========================================
# 1. إعداد جيميناي
# ==========================================
# جلب المفتاح من بيئة السيرفر (Render) لضمان الأمان
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    # تعديل اسم الموديل ليكون النسخة المستقرة
    model_llm = genai.GenerativeModel('gemini-2.5-flash') 

# ==========================================
# 2. تهيئة التطبيق (FastAPI)
# ==========================================
app = FastAPI(title="Roawiah AI Engine")

# تفعيل CORS للسماح لموقع فرح بالاتصال بالسيرفر بدون قيود متصفح
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 3. تحميل الموديلات والمشفرات
# ==========================================
try:
    rf_model = joblib.load('emotional_model.pkl')
    le_cat = joblib.load('category_encoder.pkl')
    le_brand = joblib.load('brand_encoder.pkl')
    print("✅ تم تحميل الموديلات بنجاح!")
except Exception as e:
    print(f"❌ خطأ في تحميل الموديلات: {e}")

# ==========================================
# 4. تعريف هيكل البيانات (المدخلات من موقع فرح)
# ==========================================
class PurchaseRequest(BaseModel):
    hour: int
    day_of_week: int
    main_category: str
    brand: str
    product_name: str 

# ==========================================
# 5. البوابة الرئيسية لمعالجة الطلبات
# ==========================================
@app.post("/predict")
async def predict_emotion(request: PurchaseRequest):
    try:
        # أ. تحويل النصوص (Category & Brand) إلى أرقام يفهمها الموديل
        cat_enc = le_cat.transform([request.main_category])[0]
        brand_enc = le_brand.transform([request.brand])[0]
        
        # ب. التنبؤ بالتهور بناءً على الميزات والوقت
        features = [[request.hour, request.day_of_week, cat_enc, brand_enc]]
        prob = rf_model.predict_proba(features)[0][1]
        
        # استخدام العتبة 60% كما حددتِ في تجاربك
        prediction = 1 if prob >= 0.60 else 0
        is_emotional = bool(prediction == 1)
        
        # ج. صياغة الرسالة عبر Gemini في حالة الاندفاع فقط
        gemini_message = ""
        if is_emotional and GEMINI_API_KEY:
            tone = "نصيحة أخوية لطيفة جداً" if prob < 0.8 else "تنبيه حريص وودي"
            prompt = f"""
            أنت مساعد ذكي لموقع تسوق أردني راقي.
            المهمة: كتابة رسالة قصيرة جداً لمستخدم يفكر في شراء {request.product_name}.
            الأسلوب المطلوب: "لطيف" (لغة بيضاء مهذبة)، مناسب للذكور والإناث.
            النبرة: {tone}.
            الهدف: اقتراح التفكير لمدة دقيقتين قبل إتمام الدفع لضمان الرضا عن القرار.
            """
            response = model_llm.generate_content(prompt)
            gemini_message = response.text.strip()
        
        # د. إرسال الرد النهائي
        return {
            "is_emotional": is_emotional,
            "probability": prob,
            "message": gemini_message
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
