#!/usr/bin/env python3
"""Generate synthetic training data for structured extraction."""
import json
import random
import hashlib
from datetime import datetime

random.seed(42)

TIMESTAMP = "2026-06-16T00:00:00Z"
prompt_counter = [0]

def get_prompt_id():
    prompt_counter[0] += 1
    return f"gen_{prompt_counter[0]:04d}"

def meta():
    return {
        "generation_model": "kiro-generated",
        "generation_timestamp": TIMESTAMP,
        "prompt_id": get_prompt_id()
    }

# ============ CONFERENCE TALK DATA ============
SPEAKERS = [
    "Dr. Sarah Chen", "Prof. Marcus Williams", "Aisha Patel", "Dr. Tomás García",
    "Prof. Yuki Tanaka", "Dr. Oluwaseun Adeyemi", "Maria Kowalski", "Dr. James O'Brien",
    "Prof. Fatima Al-Rashid", "Dr. Rajesh Kumar", "Elena Volkov", "Dr. Michael Torres",
    "Prof. Ingrid Bergström", "Dr. Chiara Rossi", "Kwame Asante", "Dr. Liam Murphy",
    "Prof. Mei-Lin Zhang", "Dr. Sofia Papadopoulos", "André Dubois", "Dr. Priya Sharma",
    "Prof. Henrik Nielsen", "Dr. Amara Okafor", "Juan Carlos Mendoza", "Dr. Rachel Kim",
    "Prof. Lars Eriksson", "Dr. Nadia Hassan", "Roberto Fernandez", "Dr. Emily Watson",
    "Prof. Hiroshi Yamamoto", "Dr. Cynthia Blackwell", "Omar Khalid", "Dr. Anna Petrov",
    "Prof. David Nakamura", "Dr. Isabel Santos", "Benjamin Wright", "Dr. Anika Desai",
    "Prof. Thomas Müller", "Dr. Grace Okonkwo", "Pierre Leblanc", "Dr. Sunita Rao",
    "Prof. Kenji Watanabe", "Dr. Clara Magnusson", "Ahmed Zayed", "Dr. Victoria Hayes",
    "Prof. Dimitri Sokolov", "Dr. Angela Martínez", "Patrick O'Sullivan", "Dr. Leila Nazari",
    "Prof. Gunther Schmidt", "Dr. Blessing Eze"
]

TOPICS = [
    "Transformer Architectures for Edge Computing", "Federated Learning in Healthcare",
    "Quantum Error Correction", "Sustainable AI Infrastructure", "Privacy-Preserving ML",
    "Graph Neural Networks for Drug Discovery", "Autonomous Vehicle Navigation",
    "Natural Language Understanding in Low-Resource Languages", "Reinforcement Learning for Robotics",
    "Explainable AI in Clinical Decision-Making", "Zero-Shot Learning Advances",
    "Computer Vision for Agricultural Monitoring", "Adversarial Robustness in Deep Networks",
    "Efficient Fine-Tuning of Large Language Models", "Causal Inference in Observational Studies",
    "Multi-Modal Learning for Accessibility", "Blockchain-Based Data Provenance",
    "Neuromorphic Computing Architectures", "Ethical AI Governance Frameworks",
    "Time Series Forecasting with Attention Mechanisms", "Knowledge Graphs for Scientific Discovery",
    "Continual Learning Without Catastrophic Forgetting", "Protein Structure Prediction",
    "Automated Theorem Proving", "AI-Driven Material Discovery",
    "Generative Models for Synthetic Data", "Edge AI for IoT Deployments",
    "Bias Mitigation in Hiring Algorithms", "Speech Recognition in Noisy Environments",
    "Neural Architecture Search", "Diffusion Models for Image Generation",
    "Climate Modeling with Machine Learning", "Cybersecurity Threat Detection with AI",
    "Human-Robot Interaction Design", "Recommendation Systems at Scale",
    "Medical Image Segmentation", "AI for Code Generation and Review",
    "Distributed Training Optimization", "Fairness in Criminal Justice Algorithms",
    "Smart Grid Optimization with Deep Learning"
]

CONFERENCES = [
    "NeurIPS 2025", "ICML 2025", "ACL 2025", "CVPR 2025", "AAAI 2025",
    "ICLR 2025", "KDD 2025", "SIGIR 2025", "EMNLP 2025", "ECCV 2025",
    "IEEE Big Data 2025", "WSDM 2025", "RecSys 2025", "NAACL 2025",
    "International Conference on Robotics and Automation 2025",
    "ACM CHI Conference 2025", "Web Conference 2025", "IJCAI 2025",
    "AISTATS 2025", "UAI 2025", "CoRL 2025", "MICCAI 2025",
    "IEEE Symposium on Security and Privacy 2025", "VLDB 2025",
    "International Conference on Machine Learning and Applications",
    "European Conference on Computer Vision", "Pacific Graphics 2025",
    "Interspeech 2025", "SIGGRAPH 2025", "International Conference on Data Mining",
    "Global AI Summit 2025", "AI Ethics Symposium 2025",
    "Healthcare AI Conference 2025", "FinTech AI Forum 2025",
    "Climate Tech Summit 2025", "Quantum Computing Conference 2025",
    "DevOps World 2025", "PyCon 2025", "RustConf 2025", "JSConf EU 2025"
]

LOCATIONS = [
    "San Francisco, CA, USA", "Vancouver, Canada", "London, UK", "Berlin, Germany",
    "Tokyo, Japan", "Singapore", "Sydney, Australia", "Paris, France",
    "New York City, USA", "Toronto, Canada", "Zurich, Switzerland", "Barcelona, Spain",
    "Seoul, South Korea", "Amsterdam, Netherlands", "Vienna, Austria",
    "Stockholm, Sweden", "Dubai, UAE", "Cape Town, South Africa",
    "São Paulo, Brazil", "Mumbai, India", "Beijing, China", "Tel Aviv, Israel",
    "Helsinki, Finland", "Dublin, Ireland", "Montreal, Canada",
    "Los Angeles, CA, USA", "Boston, MA, USA", "Seattle, WA, USA",
    "Austin, TX, USA", "Chicago, IL, USA", "Washington DC, USA",
    "Lisbon, Portugal", "Prague, Czech Republic", "Taipei, Taiwan",
    "Bangkok, Thailand", "Nairobi, Kenya", "Buenos Aires, Argentina",
    "Oslo, Norway", "Edinburgh, UK", "Kyoto, Japan"
]

TALK_TEMPLATES_SIMPLE = [
    "At {conf} in {loc}, {speaker} delivered a fascinating presentation on {topic}. The talk covered recent advances and practical applications in the field. Attendees particularly appreciated the live demonstrations and real-world case studies that were shared throughout the session.",
    "{speaker} took the stage at {conf}, held this year in {loc}, to discuss {topic}. Their presentation was one of the most well-attended sessions of the entire conference. The Q&A session afterward ran over the allotted time due to intense audience interest.",
    "The {conf} conference in {loc} featured an outstanding keynote by {speaker} about {topic}. Drawing on years of research and industry experience, the speaker presented compelling evidence for new approaches in this domain.",
    "During {conf} ({loc}), {speaker} presented their latest work on {topic}. The session attracted researchers and practitioners from around the world. Several attendees noted it was the highlight of the conference.",
    "{speaker} recently spoke at {conf} in {loc}. The topic of their talk was {topic}, which has been gaining significant attention in the research community. The presentation included novel experimental results and a roadmap for future work.",
    "One of the standout presentations at {conf} this year was by {speaker}, who spoke about {topic}. The conference was held in {loc} and drew thousands of participants from academia and industry alike.",
    "I attended a great talk by {speaker} at {conf} in {loc}. They presented on {topic} and shared some really interesting findings. The audience was engaged throughout and there were lots of great questions afterward.",
    "This year's {conf} was held in {loc}. Among the speakers was {speaker}, who gave a compelling talk about {topic}. Their presentation style was clear and accessible, making complex ideas understandable for the diverse audience.",
    "{speaker} presented at {conf} in {loc} on the topic of {topic}. The research presented showed promising results that could have significant impact on the field. Multiple industry partners expressed interest in collaborating on follow-up work.",
    "At the {conf} venue in {loc}, {speaker} shared groundbreaking research on {topic}. The packed auditorium was a testament to the relevance and timeliness of this research area.",
]

TALK_TEMPLATES_ADVERSARIAL = [
    "I think it was {speaker} - or maybe it was someone else from their lab - who gave a talk at what I believe was {conf} in {loc}. They talked about something related to {topic}, though the actual title might have been slightly different. The details are a bit fuzzy since I attended so many sessions that day.",
    "{speaker} was scheduled to present on {topic} at {conf} in {loc}, but due to travel complications, the talk was delivered virtually. Meanwhile, Dr. Someone Else presented on a completely unrelated topic about underwater basket weaving algorithms in the adjacent room. The conference also featured a panel on AI ethics with five other speakers.",
    "So yesterday I was at {conf} - beautiful venue in {loc} by the way, the food was amazing and the hotel had this gorgeous rooftop pool - anyway, {speaker} gave this incredible talk on {topic}. Oh wait, I should mention that the coffee breaks were unusually long, and there was this interesting conversation I had with someone about parking. But yeah, great talk!",
    "The program listed {speaker} as presenting on {topic} at {conf} ({loc}), however some sources indicate the talk was actually co-presented with another researcher. Additionally, there's conflicting information about whether this was a full presentation or a poster session, and some attendees recall the topic being slightly different from what was advertised.",
]

def generate_conference_talk(difficulty="simple", adversarial=False):
    speaker = random.choice(SPEAKERS)
    topic = random.choice(TOPICS)
    conf = random.choice(CONFERENCES)
    loc = random.choice(LOCATIONS)
    
    if adversarial:
        template = random.choice(TALK_TEMPLATES_ADVERSARIAL)
        difficulty = "complex"
    else:
        template = random.choice(TALK_TEMPLATES_SIMPLE)
    
    input_text = template.format(speaker=speaker, topic=topic, conf=conf, loc=loc)
    
    expected_output = {
        "speaker_name": speaker,
        "topic": topic,
        "conference": conf,
        "location": loc
    }
    
    return {
        "input_text": input_text,
        "expected_output": expected_output,
        "schema_id": "conference_talk_simple",
        "difficulty_level": difficulty,
        "source_metadata": meta()
    }

# ============ PRODUCT LISTING DATA ============
PRODUCT_NAMES = [
    "UltraFit Pro Wireless Earbuds", "NaturGlow Vitamin C Serum", "TechMaster 4K Monitor",
    "CloudWalk Memory Foam Sneakers", "SmartBrew Coffee Maker Pro", "AeroGlide Standing Desk",
    "PureBlend Protein Powder", "CrystalClear Water Filter", "FlexiGrip Yoga Mat",
    "ThunderBolt Gaming Mouse", "SilkTouch Bamboo Sheets Set", "PowerVault Battery Pack",
    "GreenThumb Smart Planter", "SpeedDemon Racing Helmet", "LumiGlow LED Desk Lamp",
    "OceanBreeze Diffuser", "TitanForce Resistance Bands", "SwiftType Mechanical Keyboard",
    "NovaStar Telescope", "PetPal Automatic Feeder", "AquaPure Shower Filter",
    "ZenMind Meditation Cushion", "TurboChef Air Fryer XL", "QuantumLeap SSD 2TB",
    "SolarPeak Portable Charger", "MicroFiber Ultra Mop System", "SoundScape Noise Machine",
    "FrostBite Insulated Bottle", "PixelPerfect Drawing Tablet", "BreezeKing Tower Fan",
    "IronClad Phone Case", "DreamWeave Weighted Blanket", "SparkCharge EV Charger",
    "HydroFlow Garden Hose", "CoreStrength Ab Roller", "VisionMax Blue Light Glasses",
    "ChefElite Knife Set", "ComfortZone Heated Blanket", "TrailBlazer Hiking Backpack",
    "CodeCraft Developer Board"
]

BRANDS = [
    "TechNova", "NaturalElements", "PeakPerformance", "HomeHaven", "FitLife",
    "QuantumTech", "GreenWave", "SwiftGear", "PureLiving", "CloudNine",
    "ThunderForce", "EcoSmart", "PowerHouse", "AquaVita", "StellarTech",
    "VitalCore", "Nexus", "PrimePath", "EliteEdge", "ZenithLabs",
    "SkyBound", "OmniTech", "FlowState", "BrightSide", "CoreLogic",
    "WaveRider", "SunPeak", "MountainTop", "CrystalTech", "PureForce"
]

CATEGORIES = ["electronics", "clothing", "home_garden", "sports_outdoors", "books",
              "food_beverage", "health_beauty", "toys_games", "automotive", "other"]

CONDITIONS = ["new", "refurbished", "used_like_new", "used_good", "used_acceptable"]
AVAILABILITIES = ["in_stock", "out_of_stock", "pre_order", "limited_stock"]
CURRENCIES = ["USD", "EUR", "GBP", "CAD", "AUD", "JPY"]

PRODUCT_FEATURES = [
    "Bluetooth 5.3 connectivity", "IPX7 water resistance", "USB-C fast charging",
    "Ergonomic design", "Medical-grade materials", "Energy Star certified",
    "Made from recycled materials", "Dishwasher safe", "Machine washable",
    "5-year warranty included", "Adjustable height settings", "LED display",
    "Voice control compatible", "Anti-slip base", "Foldable design",
    "Temperature control", "Auto-shutoff feature", "Noise cancellation",
    "Quick-release mechanism", "UV protection coating", "Scratch resistant surface",
    "BPA-free materials", "Dual motor system", "Wireless charging pad",
    "Smart app integration", "360-degree rotation", "Memory foam padding",
    "Stainless steel construction", "Non-toxic finish", "Modular components"
]

PRODUCT_TEMPLATES_SIMPLE = [
    "Introducing the {product} by {brand}. This {category_text} item features {features_text}. Currently priced at {price_text}, it comes in {condition} condition. Dimensions are {dims_text}. {avail_text}. SKU: {sku}.",
    "Check out the {brand} {product}! It's a {category_text} product that offers {features_text}. You can get it for {price_text}. The item is {condition} and measures {dims_text}. {avail_text}. Product code: {sku}.",
    "The {product} from {brand} is now available in our {category_text} section. Key features include {features_text}. Priced at {price_text}, this {condition} item has dimensions of {dims_text}. {avail_text}. Reference: {sku}.",
    "Looking for a great {category_text} product? The {brand} {product} might be just what you need. It offers {features_text}. The current price is {price_text} for a {condition} unit. Size: {dims_text}. {avail_text}. Item number: {sku}.",
    "{brand} presents the {product}, a premium {category_text} offering. Notable features: {features_text}. Available at {price_text} in {condition} condition. Package dimensions: {dims_text}. {avail_text}. Catalog ID: {sku}.",
]

PRODUCT_TEMPLATES_ADVERSARIAL = [
    "So I found this {product} by {brand} at what I think was {price_text} - or maybe that was the price of something else I was looking at. It's supposed to be a {category_text} item with {features_text}, but honestly the listing was confusing. The dimensions said {dims_text} but the picture looked way bigger. Oh and there was also this other product from a different brand right next to it that looked almost identical. Condition listed as {condition} but one review said it arrived damaged. {avail_text}. SKU might be {sku} but don't quote me on that.",
    "FLASH SALE - BUY NOW - LIMITED TIME OFFER!!! The amazing incredible {brand} {product}!!! This REVOLUTIONARY {category_text} product will CHANGE YOUR LIFE with {features_text}. Was originally $9999.99 but NOW only {price_text}!!! Dimensions: {dims_text}. Condition: {condition}. {avail_text}. Act fast! SKU: {sku}. Also check out our other deals on unrelated products like garden hoses and vintage postcards!",
    "The {brand} {product} - a {category_text} item - is described as having {features_text}. However, there's conflicting information: some sources say the price is {price_text} while others list it at a completely different amount. The condition is listed as {condition} on one page but as something else on the product detail page. Dimensions given as {dims_text}. {avail_text}. SKU: {sku}. Note: this listing may be outdated.",
]

def generate_sku():
    return f"{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}-{random.randint(1000,9999)}-{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.randint(10,99)}"

def generate_product_listing(difficulty="medium", adversarial=False):
    product = random.choice(PRODUCT_NAMES)
    brand = random.choice(BRANDS)
    category = random.choice(CATEGORIES)
    condition = random.choice(CONDITIONS)
    availability = random.choice(AVAILABILITIES)
    currency = random.choice(CURRENCIES)
    price_amount = round(random.uniform(9.99, 999.99), 2)
    discount = random.choice([0, 5, 10, 15, 20, 25, 30, 50]) if random.random() > 0.4 else 0
    length_cm = round(random.uniform(5, 120), 1)
    width_cm = round(random.uniform(3, 80), 1)
    height_cm = round(random.uniform(1, 60), 1)
    weight_kg = round(random.uniform(0.1, 25.0), 2)
    features = random.sample(PRODUCT_FEATURES, random.randint(2, 5))
    sku = generate_sku()
    
    category_text = category.replace("_", " & ") if "_" in category else category
    features_text = ", ".join(features)
    price_text = f"{price_amount} {currency}" + (f" ({discount}% off)" if discount else "")
    dims_text = f"{length_cm}cm x {width_cm}cm x {height_cm}cm, weighing {weight_kg}kg"
    avail_text = availability.replace("_", " ").title()
    
    if adversarial:
        template = random.choice(PRODUCT_TEMPLATES_ADVERSARIAL)
        difficulty = "complex"
    else:
        template = random.choice(PRODUCT_TEMPLATES_SIMPLE)
    
    input_text = template.format(
        product=product, brand=brand, category_text=category_text,
        features_text=features_text, price_text=price_text, dims_text=dims_text,
        condition=condition, avail_text=avail_text, sku=sku
    )
    
    description = f"A {condition} {category_text} product by {brand} featuring {features[0].lower()}"
    
    expected_output = {
        "product_name": product,
        "brand": brand,
        "category": category,
        "description": description,
        "pricing": {
            "amount": price_amount,
            "currency": currency,
            "discount_percentage": discount
        },
        "dimensions": {
            "length_cm": length_cm,
            "width_cm": width_cm,
            "height_cm": height_cm,
            "weight_kg": weight_kg
        },
        "condition": condition,
        "availability": availability,
        "sku": sku,
        "features": features
    }
    
    return {
        "input_text": input_text,
        "expected_output": expected_output,
        "schema_id": "product_listing_medium",
        "difficulty_level": difficulty,
        "source_metadata": meta()
    }

# ============ SCIENTIFIC PAPER DATA ============
PAPER_TITLES = [
    "Efficient Attention Mechanisms for Long-Sequence Modeling",
    "A Novel Framework for Multi-Agent Cooperative Learning",
    "Deep Learning Approaches to Protein Folding Prediction",
    "Scalable Graph Neural Networks for Molecular Property Prediction",
    "Federated Learning with Differential Privacy Guarantees",
    "Causal Discovery in High-Dimensional Time Series Data",
    "Robust Object Detection Under Adversarial Conditions",
    "Self-Supervised Pre-Training for Low-Resource NLP",
    "Quantum-Classical Hybrid Algorithms for Optimization",
    "Neural Radiance Fields for Dynamic Scene Reconstruction",
    "Interpretable Machine Learning for Clinical Risk Assessment",
    "Efficient Training of Sparse Mixture-of-Experts Models",
    "Zero-Shot Cross-Lingual Transfer with Language Adapters",
    "Physics-Informed Neural Networks for Fluid Dynamics",
    "Continual Learning in Non-Stationary Environments",
    "Generative Adversarial Networks for Drug Molecule Design",
    "Transformer-Based Architectures for Audio Processing",
    "Multi-Task Learning with Gradient Balancing",
    "Reinforcement Learning for Autonomous Drone Navigation",
    "Fairness-Aware Classification with Noisy Labels",
    "Attention-Based Models for Medical Image Analysis",
    "Probabilistic Programming for Bayesian Deep Learning",
    "Few-Shot Learning via Meta-Learning Optimization",
    "Diffusion Models for High-Resolution Image Synthesis",
    "Knowledge Distillation for Edge Deployment",
    "Robust Speech Recognition in Multi-Speaker Settings",
    "Graph Transformers for Chemical Reaction Prediction",
    "Privacy-Preserving Federated Recommendation Systems",
    "Neural Architecture Search with Hardware Constraints",
    "Self-Supervised Learning for Point Cloud Understanding",
    "Large Language Models for Mathematical Reasoning",
    "Temporal Fusion Transformers for Financial Forecasting",
    "Adversarial Training Strategies for Vision Transformers",
    "Multi-Modal Fusion for Autonomous Driving",
    "Efficient Inference on Resource-Constrained Devices",
    "Active Learning with Human-in-the-Loop Feedback",
    "Domain Adaptation for Satellite Image Classification",
    "Neural Ordinary Differential Equations for System Modeling",
    "Contrastive Learning for Dense Retrieval",
    "Explainable AI for Regulatory Compliance"
]

AFFILIATIONS = [
    "MIT Computer Science and AI Laboratory", "Stanford University",
    "Google DeepMind", "Carnegie Mellon University", "UC Berkeley",
    "Oxford University", "ETH Zurich", "Max Planck Institute",
    "Microsoft Research", "Meta AI Research", "Harvard University",
    "Princeton University", "University of Toronto", "Tsinghua University",
    "Cambridge University", "Georgia Tech", "University of Washington",
    "Columbia University", "Imperial College London", "EPFL",
    "University of Michigan", "Cornell University", "NYU",
    "University of Illinois Urbana-Champaign", "Caltech",
    "University of Edinburgh", "Technical University of Munich",
    "INRIA", "National University of Singapore", "Seoul National University",
    "University of Tokyo", "Peking University", "Allen Institute for AI",
    "NVIDIA Research", "Apple ML Research", "Amazon Science",
    "Johns Hopkins University", "Duke University", "UCLA",
    "University of Maryland"
]

VENUES = [
    "Nature Machine Intelligence", "JMLR", "IEEE TPAMI",
    "Proceedings of NeurIPS 2025", "ICML 2025 Proceedings",
    "ACL Anthology", "CVPR 2025", "Science Robotics",
    "Nature Communications", "AAAI 2025", "ICLR 2025",
    "Artificial Intelligence Journal", "Neural Computation",
    "Pattern Recognition", "Machine Learning Journal",
    "Journal of Chemical Information and Modeling",
    "Bioinformatics", "Physical Review Letters",
    "PNAS", "Cell Systems", "IEEE Transactions on Neural Networks",
    "ACM Computing Surveys", "Transactions on Graphics",
    "Journal of Artificial Intelligence Research",
    "Frontiers in Computational Neuroscience"
]

RESEARCH_FIELDS = ["computer_science", "biology", "physics", "chemistry", "mathematics",
                   "medicine", "engineering", "social_sciences", "environmental_science",
                   "materials_science", "other"]

METHODOLOGIES = ["experimental", "computational", "theoretical", "survey",
                 "meta_analysis", "case_study", "mixed_methods"]

KEYWORDS_POOL = [
    "deep learning", "neural networks", "machine learning", "optimization",
    "attention mechanism", "transformer", "graph neural networks", "reinforcement learning",
    "natural language processing", "computer vision", "generative models",
    "federated learning", "transfer learning", "self-supervised learning",
    "explainability", "fairness", "privacy", "efficiency", "scalability",
    "robustness", "few-shot learning", "meta-learning", "multi-task learning",
    "domain adaptation", "knowledge distillation", "pruning", "quantization",
    "distributed training", "edge computing", "IoT", "healthcare AI",
    "drug discovery", "protein folding", "climate modeling", "autonomous systems",
    "speech processing", "recommendation systems", "information retrieval",
    "causal inference", "Bayesian methods", "variational inference",
    "contrastive learning", "diffusion models", "neural architecture search"
]

DATASETS = [
    ("ImageNet-1K", "Stanford Vision Lab", "1.2M images"),
    ("COCO 2017", "Microsoft", "330K images"),
    ("WikiText-103", "Salesforce", "103M tokens"),
    ("OpenWebText", "OpenAI community", "8M documents"),
    ("PubMed Central", "NIH", "4.5M articles"),
    ("ChEMBL", "EMBL-EBI", "2.1M compounds"),
    ("MIMIC-III", "MIT Lab for Computational Physiology", "58K ICU stays"),
    ("Common Crawl", "Common Crawl Foundation", "250B pages"),
    ("LibriSpeech", "OpenSLR", "1000 hours"),
    ("MS MARCO", "Microsoft", "1M queries"),
    ("SQuAD 2.0", "Stanford NLP", "150K questions"),
    ("GLUE Benchmark", "NYU/DeepMind", "9 tasks"),
    ("Cityscapes", "Daimler AG", "25K images"),
    ("ModelNet40", "Princeton", "12K CAD models"),
    ("QM9", "Technical University of Berlin", "134K molecules"),
    ("ZINC", "University of California", "750M compounds"),
    ("Kinetics-700", "DeepMind", "650K videos"),
    ("AudioSet", "Google", "2M clips"),
    ("LAION-5B", "LAION", "5.85B image-text pairs"),
    ("The Pile", "EleutherAI", "800GB text")
]

FUNDING = [
    "National Science Foundation (NSF)", "European Research Council (ERC)",
    "DARPA", "Google Research Grant", "Microsoft Research Award",
    "NIH R01 Grant", "EPSRC", "German Research Foundation (DFG)",
    "Canadian NSERC", "Swiss National Science Foundation",
    "Meta Research Grant", "Amazon Science Hub", "NVIDIA Academic Grant",
    "Toyota Research Institute", "Samsung AI Center",
    "Bill and Melinda Gates Foundation", "Wellcome Trust",
    "Simons Foundation", "Howard Hughes Medical Institute",
    "Chan Zuckerberg Initiative"
]

PAPER_TEMPLATES = [
    "A recent paper titled \"{title}\" by {authors_text} was published in {venue}. The study, dated {date}, uses a {methodology} approach within the field of {field_text}. The abstract states: \"{abstract}\" Key findings include: {findings_text}. The research utilized {datasets_text}. Keywords: {keywords_text}. This peer-reviewed work was funded by {funding_text}. The code is available at {repo}. DOI: {doi}. It has been cited {citations} times so far.",
    "Published in {venue} on {date}, the paper \"{title}\" presents novel work in {field_text}. Authors {authors_text} employed {methodology} methods to investigate their research questions. Their abstract reads: \"{abstract}\" The team reported several important findings: {findings_text}. Datasets used included {datasets_text}. The work was supported by {funding_text} and the code repository is at {repo}. DOI: {doi}. Citation count: {citations}. Keywords include {keywords_text}.",
    "The paper \"{title}\" ({doi}), authored by {authors_text}, represents a significant contribution to {field_text}. Published in {venue} ({date}), this {methodology} study addresses key challenges in the domain. Abstract: \"{abstract}\" Notable findings: {findings_text}. The researchers worked with {datasets_text} and acknowledged support from {funding_text}. Source code: {repo}. The paper has accumulated {citations} citations. Relevant keywords: {keywords_text}.",
    "In their {methodology} study published in {venue}, {authors_text} present \"{title}\" - a comprehensive investigation in {field_text}. The publication date is {date} and the DOI is {doi}. From the abstract: \"{abstract}\" Key results include {findings_text}. The analysis was performed on {datasets_text}. Funding acknowledgments go to {funding_text}. Implementation details can be found at {repo}. Current citation count stands at {citations}. Paper keywords: {keywords_text}.",
]

PAPER_TEMPLATES_ADVERSARIAL = [
    "I came across this paper - I think it was called \"{title}\" or something similar - by {authors_text}, maybe published in {venue} around {date}. The DOI might be {doi} but I'm not 100% sure. It seemed to be about {field_text} using {methodology} methods. I remember the abstract mentioned something about \"{abstract}\" but I might be confusing it with another paper I read the same day. They found {findings_text}, or at least that's what I took away from it. Datasets: {datasets_text}. I heard it was funded by {funding_text}. Code supposedly at {repo} but the link might be broken. Keywords: {keywords_text}. Citations: {citations} last I checked but that changes daily. Oh and there was also this other paper on a completely different topic by some of the same authors.",
    "BREAKING: Revolutionary paper \"{title}\" SHATTERS previous records! {authors_text} have published what experts are calling a GAME-CHANGING study in {venue} ({date}). Using cutting-edge {methodology} techniques in {field_text}, they prove beyond doubt that {findings_text}. Abstract: \"{abstract}\" - NOTE: the original abstract was 3 pages long, this is heavily summarized. Utilized {datasets_text} plus several proprietary datasets not listed. DOI: {doi}. Funded by {funding_text} and possibly others. Code: {repo}. {citations} citations and growing RAPIDLY! Keywords: {keywords_text}. MUST READ!",
    "Paper details - partially verified: Title: \"{title}\" (might have been updated since publication). Authors: {authors_text} - note that author order was disputed and a correction was later issued. Venue: {venue}, {date}. DOI: {doi}. Field: {field_text}. Methodology: listed as {methodology} but reviewers questioned whether it truly qualifies. Abstract excerpt: \"{abstract}\" Claimed findings: {findings_text}. Datasets: {datasets_text} - though data availability has been questioned. Funding: {funding_text}. Repository: {repo} (last commit was 2 years ago). Citations: {citations}. Keywords: {keywords_text}.",
]

ABSTRACTS = [
    "We propose a novel architecture that achieves state-of-the-art performance while reducing computational requirements by 40%. Our approach leverages hierarchical attention patterns to efficiently process long sequences without sacrificing accuracy.",
    "This paper introduces a unified framework for multi-agent learning that enables cooperative behavior in complex environments. We demonstrate significant improvements over existing baselines across multiple benchmark tasks.",
    "We present a comprehensive study of privacy-preserving techniques in distributed machine learning. Our theoretical analysis provides formal guarantees while our experiments show minimal performance degradation.",
    "Our work addresses the challenge of training deep networks with limited labeled data. We propose a self-supervised pre-training strategy that transfers effectively to downstream tasks across multiple domains.",
    "We investigate the fundamental limitations of current approaches and propose theoretically-grounded solutions. Our analysis reveals important connections between optimization landscapes and generalization properties.",
    "This study presents a scalable method for processing graph-structured data that achieves linear complexity with respect to graph size. We validate our approach on datasets ranging from molecular graphs to social networks.",
    "We introduce a new benchmark and evaluation framework for assessing model robustness under distribution shift. Our findings highlight critical vulnerabilities in current methods and propose effective mitigations.",
    "Our research demonstrates that carefully designed training curricula can significantly improve sample efficiency in reinforcement learning. We provide both theoretical justification and extensive empirical validation.",
    "We propose a hardware-aware optimization technique that enables deployment of large models on edge devices. Our method achieves 10x compression with less than 1% accuracy loss.",
    "This paper establishes new theoretical bounds for learning in non-stationary environments. We complement our analysis with practical algorithms that adapt to changing data distributions.",
]

FINDINGS_POOL = [
    "achieved 94.3% accuracy on the benchmark dataset, surpassing previous SOTA by 2.1%",
    "reduced training time by 60% compared to baseline methods",
    "demonstrated robust performance under adversarial perturbations",
    "showed significant improvements in low-resource settings",
    "identified critical failure modes in existing approaches",
    "established new theoretical bounds for convergence",
    "achieved human-level performance on three evaluation tasks",
    "reduced model size by 75% with minimal accuracy loss",
    "demonstrated transferability across five different domains",
    "showed consistent improvements across all demographic groups",
    "reduced inference latency by 4x on mobile devices",
    "provided formal privacy guarantees with epsilon=0.5",
    "achieved statistical significance with p<0.001 across all experiments",
    "showed 30% improvement in sample efficiency",
    "demonstrated scalability to graphs with 100M+ nodes",
]

def generate_doi():
    return f"10.{random.randint(1000,9999)}/{random.choice(['ml','ai','cs','nn','dl'])}.{random.randint(2025000,2025999)}"

def generate_repo():
    user = random.choice(["research-lab", "mlgroup", "ai-team", "deeplearn", "neural-nets", "scicomp"])
    project = random.choice(["efficient-attention", "multi-agent-rl", "privacy-ml", "graph-nn", "edge-deploy", "robust-cv", "nlp-transfer", "bio-predict"])
    return f"https://github.com/{user}/{project}-{random.randint(2024,2025)}"

def generate_scientific_paper(difficulty="complex", adversarial=False):
    title = random.choice(PAPER_TITLES)
    num_authors = random.randint(2, 6)
    authors = []
    selected_names = random.sample(SPEAKERS, num_authors)
    selected_affiliations = random.sample(AFFILIATIONS, min(num_authors, len(AFFILIATIONS)))
    for i, name in enumerate(selected_names):
        authors.append({
            "name": name,
            "affiliation": selected_affiliations[i % len(selected_affiliations)],
            "is_corresponding": (i == 0)
        })
    
    abstract = random.choice(ABSTRACTS)
    venue = random.choice(VENUES)
    date = f"2025-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
    doi = generate_doi()
    keywords = random.sample(KEYWORDS_POOL, random.randint(3, 6))
    field = random.choice(RESEARCH_FIELDS)
    methodology = random.choice(METHODOLOGIES)
    findings = random.sample(FINDINGS_POOL, random.randint(2, 4))
    
    num_datasets = random.randint(1, 3)
    datasets = []
    for ds_name, ds_source, ds_size in random.sample(DATASETS, num_datasets):
        datasets.append({"name": ds_name, "source": ds_source, "size": ds_size})
    
    num_funding = random.randint(1, 3)
    funding = random.sample(FUNDING, num_funding)
    citations = random.randint(0, 500)
    is_peer_reviewed = random.choice([True, True, True, False])
    repo = generate_repo()
    
    num_refs = random.randint(2, 4)
    references = []
    for _ in range(num_refs):
        ref_authors_list = random.sample(SPEAKERS, random.randint(1, 3))
        references.append({
            "title": random.choice(PAPER_TITLES),
            "authors": [a for a in ref_authors_list],
            "year": random.randint(2020, 2025)
        })
    
    authors_text = ", ".join([f"{a['name']} ({a['affiliation']})" for a in authors])
    field_text = field.replace("_", " ")
    findings_text = "; ".join(findings)
    datasets_text = ", ".join([f"{d['name']} from {d['source']}" for d in datasets])
    keywords_text = ", ".join(keywords)
    funding_text = ", ".join(funding)
    
    if adversarial:
        template = random.choice(PAPER_TEMPLATES_ADVERSARIAL)
        difficulty = "complex"
    else:
        template = random.choice(PAPER_TEMPLATES)
    
    input_text = template.format(
        title=title, authors_text=authors_text, venue=venue, date=date,
        doi=doi, field_text=field_text, methodology=methodology,
        abstract=abstract, findings_text=findings_text, datasets_text=datasets_text,
        keywords_text=keywords_text, funding_text=funding_text,
        repo=repo, citations=citations
    )
    
    expected_output = {
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "publication_venue": venue,
        "publication_date": date,
        "doi": doi,
        "keywords": keywords,
        "research_field": field,
        "methodology": methodology,
        "findings": findings,
        "datasets_used": datasets,
        "funding_sources": funding,
        "citation_count": citations,
        "is_peer_reviewed": is_peer_reviewed,
        "code_repository": repo,
        "references": references
    }
    
    return {
        "input_text": input_text,
        "expected_output": expected_output,
        "schema_id": "scientific_paper_complex",
        "difficulty_level": difficulty,
        "source_metadata": meta()
    }

# ============ MAIN GENERATION LOGIC ============
def generate_all_data():
    all_examples = []
    
    # Conference Talk: 250 total (210 normal + 40 adversarial)
    for i in range(210):
        difficulty = random.choice(["simple", "simple", "simple", "medium"])
        all_examples.append(generate_conference_talk(difficulty=difficulty))
    for i in range(40):
        all_examples.append(generate_conference_talk(adversarial=True))
    
    # Product Listing: 250 total (210 normal + 40 adversarial)
    for i in range(210):
        difficulty = random.choice(["medium", "medium", "medium", "complex"])
        all_examples.append(generate_product_listing(difficulty=difficulty))
    for i in range(40):
        all_examples.append(generate_product_listing(adversarial=True))
    
    # Scientific Paper: 250 total (210 normal + 40 adversarial)
    for i in range(210):
        difficulty = random.choice(["complex", "complex", "medium"])
        all_examples.append(generate_scientific_paper(difficulty=difficulty))
    for i in range(40):
        all_examples.append(generate_scientific_paper(adversarial=True))
    
    # Shuffle within each schema group to mix difficulties
    conference_talks = [e for e in all_examples if e["schema_id"] == "conference_talk_simple"]
    product_listings = [e for e in all_examples if e["schema_id"] == "product_listing_medium"]
    papers = [e for e in all_examples if e["schema_id"] == "scientific_paper_complex"]
    
    random.shuffle(conference_talks)
    random.shuffle(product_listings)
    random.shuffle(papers)
    
    # Split: 200 train, 25 val, 25 test per schema
    train_data = conference_talks[:200] + product_listings[:200] + papers[:200]
    val_data = conference_talks[200:225] + product_listings[200:225] + papers[200:225]
    test_data = conference_talks[225:250] + product_listings[225:250] + papers[225:250]
    
    # Shuffle final splits
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    return train_data, val_data, test_data

def write_jsonl(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    import os
    
    output_dir = "/Users/bhavanar/Documents/small_model_supremacy/data"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating data...")
    train_data, val_data, test_data = generate_all_data()
    
    print(f"Train: {len(train_data)} examples")
    print(f"Val: {len(val_data)} examples")
    print(f"Test: {len(test_data)} examples")
    
    write_jsonl(train_data, os.path.join(output_dir, "train.jsonl"))
    write_jsonl(val_data, os.path.join(output_dir, "val.jsonl"))
    write_jsonl(test_data, os.path.join(output_dir, "test.jsonl"))
    
    # Verify
    for split in ["train", "val", "test"]:
        filepath = os.path.join(output_dir, f"{split}.jsonl")
        with open(filepath, 'r') as f:
            lines = f.readlines()
            print(f"\n{split}.jsonl: {len(lines)} lines")
            # Verify each line is valid JSON
            schema_counts = {}
            difficulty_counts = {}
            for line in lines:
                obj = json.loads(line)
                schema_counts[obj["schema_id"]] = schema_counts.get(obj["schema_id"], 0) + 1
                difficulty_counts[obj["difficulty_level"]] = difficulty_counts.get(obj["difficulty_level"], 0) + 1
            print(f"  Schemas: {schema_counts}")
            print(f"  Difficulties: {difficulty_counts}")
    
    print("\nDone! All files generated successfully.")
