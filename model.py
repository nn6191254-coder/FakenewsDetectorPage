import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


class NewsDetector:
    def __init__(self, dataset_path="data/news.csv"):
        self.dataset_path = Path(dataset_path)
        self.model = None
        self.metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
        self.total_samples = 0
        self._ensure_dataset_exists()
        self._train_model()

    def _ensure_dataset_exists(self):
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)

        sample_rows = [
            # =========================================================================
            # 1. LEGITIMATE / RELIABLE / NON-SPAM (label = 1)
            # =========================================================================
            # Conversational & Daily Messages
            {"text": "Hey, are you free this evening? Let us meet at the coffee shop around 6 PM.", "label": 1},
            {"text": "Can you please send me the recipe for the chocolate cake you made last week?", "label": 1},
            {"text": "I had a wonderful time at the park today with my family. The weather was sunny and pleasant.", "label": 1},
            {"text": "Let us catch up over lunch on Friday if your schedule permits.", "label": 1},
            {"text": "Thanks for helping me move the furniture yesterday, I really appreciate your support.", "label": 1},
            {"text": "Happy birthday! Wishing you a fantastic year filled with health, joy, and success.", "label": 1},
            {"text": "Do you know what time the grocery store closes tonight? I need to pick up a few ingredients.", "label": 1},
            {"text": "I finished reading the novel you recommended and really enjoyed the character development.", "label": 1},
            {"text": "The weather is very pleasant today, let us go out for a coffee.", "label": 1},
            {"text": "Can you send the project presentation by 4 PM?", "label": 1},
            {"text": "Hello Naveen, can we review the presentation slides together before the meeting tomorrow?", "label": 1},
            {"text": "I just bought a new laptop today and it works very smoothly.", "label": 1},

            # Workplace, Business & Professional Communication
            {"text": "Please find attached the quarterly financial report for review before our team meeting tomorrow morning.", "label": 1},
            {"text": "Reminder: The team standup is scheduled for 10:00 AM in Conference Room B.", "label": 1},
            {"text": "The team meeting has been moved to 10 AM on Monday.", "label": 1},
            {"text": "Thank you for submitting your project proposal. The review committee will provide feedback by next Tuesday.", "label": 1},
            {"text": "The marketing department completed the customer survey and shared the aggregate summary with team leads.", "label": 1},
            {"text": "Here are the meeting notes and action items from today's client sync.", "label": 1},
            {"text": "Please submit your timesheets by 5 PM today for payroll processing.", "label": 1},
            {"text": "Our engineering team scheduled server maintenance for midnight Sunday to apply security patches.", "label": 1},
            {"text": "Welcome to the team! Your onboarding session is scheduled for Monday at 9:30 AM.", "label": 1},

            # Standard Transactional & Service Notifications
            {"text": "Your order #12345 has been processed and will arrive by Friday. Thank you for shopping with us.", "label": 1},
            {"text": "Your appointment with Dr. Williams has been confirmed for Thursday at 2:30 PM.", "label": 1},
            {"text": "Your flight check-in is now open. Seat 14A has been assigned for your flight to Chicago.", "label": 1},
            {"text": "Your package has been delivered to your front porch. Have a great day.", "label": 1},
            {"text": "Your monthly utility statement is ready to view online. Your auto-pay is scheduled for the 15th.", "label": 1},
            {"text": "Your car service appointment is scheduled for tomorrow morning at 8:00 AM.", "label": 1},
            {"text": "The train arrives at platform 3 at 5:45 PM.", "label": 1},

            # General, Community, Sports & Tech News
            {"text": "Local authorities opened a new public library in the downtown area with over 50,000 books and free internet access.", "label": 1},
            {"text": "Apple held its annual developer conference where they introduced new software features for iPhone and Mac users.", "label": 1},
            {"text": "The home team won the championship game last night with a final score of 3 to 1 in overtime.", "label": 1},
            {"text": "Tomorrow will be partly cloudy with temperatures reaching a high of 75 degrees and light breeze.", "label": 1},
            {"text": "The city transit authority expanded bus routes to improve connectivity across residential suburbs.", "label": 1},
            {"text": "The community farmers market will feature fresh organic produce and local crafts every Saturday morning.", "label": 1},
            {"text": "Sony announced its next-generation gaming console update with improved graphics and faster loading speeds.", "label": 1},
            {"text": "The municipal water board reported that tap water quality tests met all state safety standards this quarter.", "label": 1},

            # Verified Science, Medicine & Public Policy
            {"text": "According to a study published in the New England Journal of Medicine, researchers found that regular cardiovascular exercise reduces heart disease risk by 32 percent across diverse age groups.", "label": 1},
            {"text": "The World Health Organization confirmed that global polio vaccination campaigns have reduced wild poliovirus cases by over 99 percent since 1988.", "label": 1},
            {"text": "Clinical trial results published in The Lancet demonstrate that the new mRNA malaria vaccine achieved 77 percent efficacy in pediatric trials across three West African countries.", "label": 1},
            {"text": "The Federal Reserve announced a 25 basis point interest rate adjustment following the Federal Open Market Committee meeting, citing moderation in core inflation figures.", "label": 1},
            {"text": "NASA scientists announced the discovery of a terrestrial exoplanet in the habitable zone of a nearby star system after analyzing spectrographic data from the James Webb Space Telescope.", "label": 1},
            {"text": "The Environmental Protection Agency released its annual emissions report indicating a 3.2 percent decline in nationwide greenhouse gas emissions.", "label": 1},
            {"text": "According to the Bureau of Labor Statistics, non-farm payroll employment increased by 216,000 jobs in December, while the national unemployment rate held steady at 3.7 percent.", "label": 1},
            {"text": "European Space Agency officials confirmed that the automated cargo resupply spacecraft successfully docked with the International Space Station.", "label": 1},

            # =========================================================================
            # 2. MISLEADING / CLICKBAIT / CONSPIRACY / FAKE (label = 0)
            # =========================================================================
            # Clickbait & Sensationalism
            {"text": "SHOCKING: You will never believe what doctors just admitted! This simple kitchen spice cures diabetes and cancer in 48 hours and Big Pharma is desperately trying to hide the secret from you!", "label": 0},
            {"text": "MUST READ: Famous celebrity caught red-handed in horrifying scandal that the mainstream media is desperately trying to bury from the public!", "label": 0},
            {"text": "VIRAL: Woman lost 45 pounds in just 4 days without exercise using this one weird ancient trick that nutritionists want banned immediately!", "label": 0},
            {"text": "EXCLUSIVE: Secret leaked video exposes what really happened behind closed doors at the summit! You will be horrified by the truth!", "label": 0},
            {"text": "BREAKING: Scientists reveal that eating this common everyday fruit is secretly destroying your liver and aging your skin by 20 years overnight!", "label": 0},
            {"text": "UNBELIEVABLE: Miracle water drops restore 20/20 vision in 3 days! Eye surgeons hate this man for sharing the forbidden recipe!", "label": 0},
            {"text": "WARNING: Throw away your microwave immediately! New whistleblower evidence shows it emits deadly cosmic radiation that alters human DNA!", "label": 0},
            {"text": "TOP SECRET: Billionaires are building underground bunkers because a hidden asteroid is scheduled to collide with Earth next Tuesday!", "label": 0},

            # Conspiracies & Fabricated Disinformation
            {"text": "EXCLUSIVE: Declassified military files prove that the moon landings were completely filmed on a Hollywood sound stage to trick rival nations, and secret elites are covering up the truth!", "label": 0},
            {"text": "Chemtrails revealed: Government planes are secretly spraying mind-control chemicals and synthetic pathogens over residential neighborhoods to reduce population!", "label": 0},
            {"text": "SHOCKING PROOF: 5G cell towers are transmitting subliminal biometric frequencies designed to manipulate citizen behavior and disable immune systems!", "label": 0},
            {"text": "The World Economic Forum is executing a covert master plan to ban all cash currency and replace it with microchips implanted under citizens' skin!", "label": 0},
            {"text": "Ancient pyramid texts deciphered by independent researchers prove that human civilization was engineered by reptilian extraterrestrials 5,000 years ago!", "label": 0},
            {"text": "BOMBSHELL REPORT: All weather satellites are completely fake! Earth is actually surrounded by a giant glass dome controlled by secret elites!", "label": 0},
            {"text": "Whistleblower inside the government exposes secret military weather machines causing all earthquakes and hurricanes across the globe!", "label": 0},
            {"text": "EXPOSED: Leading world leaders have already been replaced by biological clones operating from underground cloned facilities!", "label": 0},
            {"text": "Drinking pure chlorine dioxide solution 3 times a day is proven to cure autism, arthritis, and all viral infections according to suppressed holistic healers!", "label": 0},
            {"text": "Wearing face masks causes carbon dioxide poisoning and permanent brain damage within 10 minutes according to holistic freedom advocates!", "label": 0},

            # =========================================================================
            # 3. SPAM, FRAUD & PHISHING (label = 0)
            # =========================================================================
            # Prize / Lottery / Lucky Draw
            {"text": "Congratulations! You have won ₹10,00,000 in our lucky draw. Click here to claim your prize now!", "label": 0},
            {"text": "CONGRATULATIONS! You have been selected as the official lucky winner of $1,000,000 in our international lottery! Transfer $250 processing fee immediately to claim your prize now.", "label": 0},
            {"text": "Congratulations you won a free iPhone 15 Pro Max! Click this link to confirm your delivery address and pay only $10 shipping fee right now.", "label": 0},
            {"text": "You are chosen for a free gift card worth $500. Click here to claim your reward.", "label": 0},
            {"text": "URGENT: Win ₹25,000 cash bonus now. Click link to claim.", "label": 0},
            {"text": "Congratulations user! You were randomly selected for a $1,000 Walmart gift card. Complete the survey and provide your credit card to verify eligibility.", "label": 0},
            {"text": "You won! Send ₹10,000 payment to unlock your free phone delivery to your address.", "label": 0},

            # Investment / Get-Rich / Job Fraud
            {"text": "Limited-time investment opportunity! Double your money in 7 days. Join now!", "label": 0},
            {"text": "Exclusive opportunity: Make $15,000 per week working only 20 minutes a day from your phone! Send ₹5,000 registration fee to activate your automated crypto bot.", "label": 0},
            {"text": "Earn ₹5,000 daily working 1 hour from home. No experience needed. Join Telegram.", "label": 0},
            {"text": "Part time job: Earn 3000 to 5000 per day from mobile. Contact WhatsApp.", "label": 0},
            {"text": "Send 1000 rupees to receive 10000 rupees tomorrow guaranteed.", "label": 0},
            {"text": "Double your cryptocurrency in 24 hours! Send 0.1 BTC to the designated smart contract address and receive 0.2 BTC back instantly guaranteed!", "label": 0},
            {"text": "Guaranteed binary options trading profit: Deposit $200 today and withdraw $3,000 daily with our automated algorithm.", "label": 0},

            # Phishing & Account Threats
            {"text": "URGENT NOTICE: Your bank account has been flagged for suspicious activity. Click here right now and verify your online banking password and OTP to prevent immediate account termination.", "label": 0},
            {"text": "Dear customer, your electricity power will be disconnected tonight. Call officer at 9876543210.", "label": 0},
            {"text": "Your SBI Bank account is blocked due to KYC. Click link to update PAN card immediately.", "label": 0},
            {"text": "Dear user, your SIM card will be deactivated today. Call customer care immediately.", "label": 0},
            {"text": "FINAL WARNING: Your electric utility service will be permanently disconnected within 1 hour unless you immediately send payment via Bitcoin voucher.", "label": 0},
            {"text": "Pre-approved loan of ₹5,00,000 approved at 0% interest. Click here to disburse immediately.", "label": 0},
            {"text": "You received a $5,000 federal grant approved by the treasury. Send your full social security number and $100 processing charge to receive wire transfer.", "label": 0},
            {"text": "Lucky prize alert: You won ₹500,000 cash reward. Send your PAN card and bank details with ₹2,500 security deposit to release payment.", "label": 0},
            {"text": "Urgent package delivery pending: We could not deliver your parcel due to invalid address. Click here to confirm personal details and pay redelivery fee.", "label": 0},
            {"text": "Inheritance fund notification: You are named as sole beneficiary for 2.5 million dollars. Send your passport copy and release fee.", "label": 0},
        ]

        df = pd.DataFrame(sample_rows)
        df.to_csv(self.dataset_path, index=False)
        self.df = df
        self.total_samples = len(df)

    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"[^\w\s\$\₹€%]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _train_model(self):
        if self.dataset_path.exists():
            self.df = pd.read_csv(self.dataset_path)
        else:
            self._ensure_dataset_exists()

        self.total_samples = len(self.df)
        df = self.df.copy()
        df["cleaned"] = df["text"].apply(self.clean_text)
        df = df.dropna(subset=["cleaned"])

        X = df["cleaned"]
        y = df["label"].astype(int)

        train_X, test_X, train_y, test_y = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y,
        )

        self.model = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=1,
                        sublinear_tf=True,
                        stop_words="english",
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        C=2.5,
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
            ]
        )

        self.model.fit(train_X, train_y)
        predictions = self.model.predict(test_X)

        self.metrics = {
            "accuracy": round(float(accuracy_score(test_y, predictions)), 3),
            "precision": round(float(precision_score(test_y, predictions, zero_division=0)), 3),
            "recall": round(float(recall_score(test_y, predictions, zero_division=0)), 3),
            "f1": round(float(f1_score(test_y, predictions, zero_division=0)), 3),
        }

    def detect_signals(self, text):
        raw_text = str(text)
        lower = raw_text.lower()

        # 1. ATTRIBUTION & CREDIBLE SOURCES
        reliable_sources = [
            "according to", "officials said", "spokesperson", "study published in", "researchers found",
            "journal of", "published in", "confirmed by", "data shows", "statistics show", "research indicates",
            "department of", "agency announced", "organization reported", "expert says", "university",
            "reuters", "associated press", "world health organization", "centers for disease control",
            "federal reserve", "bureau of labor", "supreme court", "national oceanic", "geological survey",
            "peer-reviewed", "clinical trial", "meta-analysis", "audits confirmed", "nasa scientists",
            "environmental protection agency"
        ]

        # 2. SENSATIONAL / CLICKBAIT
        clickbait_terms = [
            "must read", "must watch", "you will never believe", "you won't believe", "shocking", "unbelievable",
            "mind blown", "blow your mind", "doctors hate", "miracle cure", "one weird trick", "cures all",
            "what happened next", "exclusive leaked", "banned immediately", "this one trick", "viral video",
            "top secret", "insane discovery", "mind blowing", "bombshell", "hidden truth", "forbidden recipe",
            "caught red-handed", "trying to bury"
        ]

        # 3. EMOTIONAL / OUTRAGE MANIPULATION
        emotional_terms = [
            "horrifying", "horrified", "furious", "terrified", "terrifying", "outrage", "deadly poison",
            "apocalypse", "nightmare", "evil plot", "atrocity", "betrayal", "disgraceful", "shameful",
            "mass panic", "hysteria", "scandal that will"
        ]

        # 4. CONSPIRACY & DISINFORMATION PATTERNS
        conspiracy_terms = [
            "declassified", "moon landing", "moon landings", "sound stage", "hollywood sound stage",
            "5g cell towers", "5g towers", "chemtrails", "synthetic pathogens", "depopulation", "deep state",
            "new world order", "illuminati", "secret elite", "secret elites", "microchip", "biological clones",
            "cloned facilities", "reptilian", "weather machine", "weather machines", "mainstream media is desperately trying",
            "mainstream media is covering", "suppressed cure", "forbidden cure", "alien coverup", "flat earth",
            "glass dome", "chlorine dioxide", "big pharma is desperately trying"
        ]

        # 5. SPAM / PHISHING / SCAM PATTERNS (REGEX POWERED)
        # Pattern 1: Prize & Lottery Claim
        p_prize = r"lucky\s*draw|lottery|won\s+.*(prize|lakh|crore|\$|\u20b9|cash|reward|iphone|gift|money|phone|mobile)|claim\s+.*(prize|reward|cash|bonus|gift|iphone|phone|mobile)|congratulations\b.*(won|winner|selected|lucky|claim|prize|mobile|phone)|\bwin\s+.*(lakh|crore|\$|\u20b9|cash|bonus|prize|reward)"
        # Pattern 2: Get-Rich / Investment / Double Asset / Fake Jobs
        p_invest = r"double\s+.*(money|crypto|cryptocurrency|bitcoin|btc|investment|funds|deposit|capital|cash)|triple\s+.*(money|crypto|funds)|investment\s+opportunity|guaranteed\s+(profit|return|income|money|payout)|earn\s+.*(daily|per\s*day|per\s*month|weekly|per\s*week|from\s*home|to\s*\d+|\u20b9|\$)|work\s+.*from\s*home|part\s*time\s*job.*(earn|daily|join|contact)|no\s+experience\s+needed.*(join|earn)|crypto\s*bot|binary\s*options|100%\s*profit|smart\s*contract.*(send|receive|address)|send\s+.*(rupees|rs|\u20b9|\$|btc|eth|usdt|bitcoin).*receive\s+.*(back|tomorrow|guaranteed|instantly)"
        # Pattern 3: Advance-Fee Fraud
        p_fee = r"(won|selected|lottery|prize|grant|beneficiary|inheritance).*(\bfee\b|processing\s*charge|registration\s*charge|security\s*deposit|transfer\s*fee|release\s*fee|shipping\s*cost|wire\s*transfer|send\s*[\$\u20b9]|send\s*btc|send\s*0\.)"
        # Pattern 4: Phishing & Account Threat
        p_phish = r"(blocked|suspended|disconnected|deactivated|compromised|flagged|power\s*cut|electricity\s*power|sim\s*card).*(verify|update|click|link|call|officer|immediately|tonight|24\s*hours|1\s*hour|today|otp|password|customer\s*care)|(kyc|pan\s*card).*(update|verify|suspend|block)"
        # Pattern 5: Pre-Approved Loan & Credit Spam
        p_loan = r"pre-?approved\s+(loan|credit|card)|instant\s+(personal\s+)?loan|0%\s*interest\s*loan|credit\s*card\s*limit\s*increased"

        is_scam = bool(
            re.search(p_prize, lower)
            or re.search(p_invest, lower)
            or re.search(p_fee, lower)
            or re.search(p_phish, lower)
            or re.search(p_loan, lower)
        )

        signals = []

        # --- Signal 1: Source & Citation Verification ---
        source_matches = [t for t in reliable_sources if t in lower]
        if len(source_matches) >= 2:
            signals.append({
                "name": "Source Attribution",
                "status": "Verified",
                "severity": "safe",
                "detail": f"Contains credible citations and institutional references (e.g., '{source_matches[0]}').",
            })
        elif len(source_matches) == 1:
            signals.append({
                "name": "Source Attribution",
                "status": "Verified",
                "severity": "safe",
                "detail": f"Contains reference to accredited source '{source_matches[0]}'.",
            })
        else:
            signals.append({
                "name": "Source Attribution",
                "status": "Direct / Clean",
                "severity": "safe",
                "detail": "Normal direct communication without external citation dependencies.",
            })

        # --- Signal 2: Clickbait & Sensationalism ---
        clickbait_matches = [t for t in clickbait_terms if t in lower]
        if len(clickbait_matches) >= 2:
            signals.append({
                "name": "Clickbait & Sensationalism",
                "status": "High Risk",
                "severity": "danger",
                "detail": f"Detected sensational trigger terms: {', '.join([f'\"{t}\"' for t in clickbait_matches[:3]])}.",
            })
        elif len(clickbait_matches) == 1:
            signals.append({
                "name": "Clickbait & Sensationalism",
                "status": "Detected",
                "severity": "danger",
                "detail": f"Found clickbait / sensational phrasing: \"{clickbait_matches[0]}\".",
            })
        else:
            signals.append({
                "name": "Clickbait & Sensationalism",
                "status": "Clear",
                "severity": "safe",
                "detail": "No sensationalist or clickbait tropes detected.",
            })

        # --- Signal 3: Emotional Manipulation ---
        emotional_matches = [t for t in emotional_terms if t in lower]
        if len(emotional_matches) >= 1:
            signals.append({
                "name": "Emotional Manipulation",
                "status": "Detected",
                "severity": "danger",
                "detail": f"Contains emotionally charged panic wording: {', '.join([f'\"{t}\"' for t in emotional_matches[:3]])}.",
            })
        else:
            signals.append({
                "name": "Emotional Manipulation",
                "status": "Neutral Tone",
                "severity": "safe",
                "detail": "Objective and balanced communicative tone.",
            })

        # --- Signal 4: Conspiracy & Disinformation ---
        conspiracy_matches = [t for t in conspiracy_terms if t in lower]
        if len(conspiracy_matches) >= 1:
            signals.append({
                "name": "Conspiracy Tropes",
                "status": "High Risk",
                "severity": "danger",
                "detail": f"Identified conspiracy narratives: {', '.join([f'\"{t}\"' for t in conspiracy_matches[:3]])}.",
            })
        else:
            signals.append({
                "name": "Conspiracy Tropes",
                "status": "Clear",
                "severity": "safe",
                "detail": "No conspiracy theory patterns or debunked tropes found.",
            })

        # --- Signal 5: Spam & Fraud Signatures ---
        if is_scam:
            signals.append({
                "name": "Spam / Fraud Signature",
                "status": "Scam Detected",
                "severity": "danger",
                "detail": "Matches financial advance-fee, lucky draw, get-rich scheme, or credential phishing signatures.",
            })
        else:
            signals.append({
                "name": "Spam / Fraud Signature",
                "status": "Clear",
                "severity": "safe",
                "detail": "No commercial fraud, lottery, or phishing patterns identified.",
            })

        # --- Signal 6: Writing Style & Quality ---
        caps_count = sum(1 for ch in raw_text if ch.isupper())
        total_letters = sum(1 for ch in raw_text if ch.isalpha())
        caps_ratio = (caps_count / total_letters) if total_letters > 0 else 0
        exclamation_count = raw_text.count("!")
        multiple_puncts = len(re.findall(r"[!?]{2,}", raw_text))

        if caps_ratio > 0.40 or exclamation_count >= 4 or multiple_puncts >= 2:
            signals.append({
                "name": "Stylometry & Quality",
                "status": "Unstable",
                "severity": "danger",
                "detail": f"Excessive capitalization ({round(caps_ratio*100)}%) or irregular punctuation clusters.",
            })
        elif caps_ratio > 0.25 or exclamation_count >= 2:
            signals.append({
                "name": "Stylometry & Quality",
                "status": "Elevated Emphasis",
                "severity": "warning",
                "detail": "Contains higher emphasis formatting.",
            })
        else:
            signals.append({
                "name": "Stylometry & Quality",
                "status": "Standard Quality",
                "severity": "safe",
                "detail": "Standard sentence structure, punctuation, and casing observed.",
            })

        return signals

    def predict_article(self, text):
        clean_t = self.clean_text(text)
        if not clean_t:
            raise ValueError("Article text is empty after cleaning.")

        # 1. ML CLASSIFIER PROBABILITY
        probabilities = self.model.predict_proba([clean_t])[0]
        classes = list(self.model.classes_)
        if 1 in classes:
            ml_reliable = float(probabilities[classes.index(1)])
        else:
            ml_reliable = float(probabilities[0])

        # 2. SIGNAL HEURISTIC SCORING
        signals = self.detect_signals(text)
        
        danger_count = sum(1 for s in signals if s.get("severity") == "danger")
        warning_count = sum(1 for s in signals if s.get("severity") == "warning")
        has_formal_source = any(
            s.get("name") == "Source Attribution" and "Verified" in s.get("status", "")
            for s in signals
        )

        # 3. FUSED RELIABILITY FORMULA
        if danger_count >= 2:
            # Severe deceptive or scam content
            reliability_score = min(0.04, ml_reliable * 0.08)
        elif danger_count == 1:
            # High risk deceptive or scam flag
            reliability_score = min(0.08, ml_reliable * 0.15)
        elif warning_count >= 2:
            # Multiple moderate red flags
            reliability_score = min(0.35, ml_reliable * 0.55)
        elif warning_count == 1:
            # Minor single warning (e.g. slight emphasis)
            reliability_score = max(0.85, min(0.92, ml_reliable))
        else:
            # ZERO RED FLAGS: Legitimate non-spam, normal messages, or verified news
            if has_formal_source:
                reliability_score = max(0.92, min(0.98, ml_reliable + 0.12))
            else:
                reliability_score = max(0.88, min(0.96, ml_reliable + 0.08))

        reliability_score = max(0.02, min(0.98, reliability_score))
        misleading_score = 1.0 - reliability_score

        # 4. FINAL VERDICT & CONFIDENCE
        if reliability_score >= 0.50:
            label = "Reliable"
            confidence = round(reliability_score * 100, 1)
        else:
            label = "Misleading"
            confidence = round(misleading_score * 100, 1)

        result = {
            "label": label,
            "confidence": confidence,
            "reliability": round(reliability_score, 3),
            "reliable_score": round(reliability_score * 100, 1),
            "misleading_score": round(misleading_score * 100, 1),
            "signals": signals,
            "metrics": self.metrics,
        }
        return result
