"""Domain lexicons used by the generator (full knowledge) and the extractor (partial knowledge).

Design note: the extractor deliberately knows only a SUBSET of the generator's phrasing,
skill relations and verbs (see HELD_OUT_* below). This reduces the circularity that would
otherwise make a synthetic benchmark trivially easy.
"""

CATEGORIES = [
    "Data Science", "Software Engineering", "DevOps & Cloud", "Cybersecurity",
    "Human Resources", "Digital Marketing", "Mechanical Engineering",
    "Civil Engineering", "Finance & Accounting", "Sales & Business Development",
]

CAT = {
    "Data Science": dict(
        core=["Python", "SQL", "Machine Learning", "Deep Learning", "NLP", "Statistics", "Pandas", "NumPy",
              "Scikit-learn", "TensorFlow", "PyTorch", "Data Visualization", "Feature Engineering",
              "Time Series", "Spark", "XGBoost"],
        secondary=["Tableau", "Power BI", "A/B Testing", "Jupyter", "Computer Vision", "Keras", "Hypothesis Testing", "Data Mining"],
        titles=["Data Analyst", "Data Scientist", "Senior Data Scientist", "Lead Data Scientist"],
        tasks=["customer churn prediction models", "recommendation engines", "demand prediction pipelines",
               "text classification services", "fraud detection models", "feature stores for analytics",
               "experimentation dashboards", "anomaly detection workflows", "sentiment analysis tools", "model health reports"],
        terms=["predictive modeling", "data-driven decision making", "statistical analysis", "model deployment"],
        certs=["Google Professional Data Engineer", "TensorFlow Developer Certificate", "IBM Data Science Professional Certificate", "Azure Data Scientist Associate"],
        degrees=[("B.Tech", "Computer Science"), ("M.Sc", "Statistics"), ("M.Tech", "Data Science"), ("B.Sc", "Mathematics")],
        projects=["Retail Demand Forecaster", "Resume Screening Classifier", "Credit Risk Scorecard", "Churn Insight Platform"]),
    "Software Engineering": dict(
        core=["Java", "Python", "JavaScript", "TypeScript", "React", "Node.js", "Spring Boot", "REST APIs",
              "Microservices", "SQL", "PostgreSQL", "MongoDB", "Git", "Data Structures", "Algorithms", "System Design"],
        secondary=["Unit Testing", "C++", "Angular", "GraphQL", "Agile", "CI/CD", "Redis", "Kafka"],
        titles=["Software Developer", "Software Engineer", "Senior Software Engineer", "Lead Software Engineer"],
        tasks=["backend services for order management", "responsive web applications", "payment gateway integrations",
               "high-traffic REST services", "internal developer tools", "event-driven processing modules",
               "customer-facing portals", "scalable API layers", "mobile backend components", "search and filtering features"],
        terms=["full-stack development", "scalable architecture", "object-oriented design", "code quality"],
        certs=["Oracle Certified Java Programmer", "AWS Certified Developer Associate", "Meta Front-End Developer Certificate", "MongoDB Developer Certification"],
        degrees=[("B.Tech", "Computer Science"), ("B.E.", "Information Technology"), ("M.Tech", "Software Engineering"), ("B.Sc", "Computer Science")],
        projects=["Inventory Management Portal", "Realtime Chat Platform", "E-commerce Backend Suite", "Booking Engine"]),
    "DevOps & Cloud": dict(
        core=["AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "Ansible", "Jenkins", "CI/CD", "Linux",
              "Bash", "Prometheus", "Grafana", "Helm", "Infrastructure as Code", "Site Reliability"],
        secondary=["Python", "Git", "Networking", "CloudFormation", "Log Management", "GitOps", "Nginx", "ELK Stack"],
        titles=["DevOps Associate", "DevOps Engineer", "Senior DevOps Engineer", "Lead Cloud Architect"],
        tasks=["automated deployment pipelines", "container orchestration platforms", "multi-region cloud infrastructure",
               "observability stacks", "zero-downtime release workflows", "cost optimization programs",
               "disaster recovery setups", "self-service environments", "configuration management frameworks", "autoscaling clusters"],
        terms=["cloud infrastructure", "release automation", "reliability engineering", "platform engineering"],
        certs=["AWS Certified Solutions Architect", "Certified Kubernetes Administrator", "HashiCorp Terraform Associate", "Azure DevOps Engineer Expert"],
        degrees=[("B.Tech", "Computer Science"), ("B.E.", "Electronics"), ("M.Tech", "Cloud Computing"), ("B.Sc", "Information Technology")],
        projects=["Container Platform Migration", "Cloud Landing Zone", "Release Pipeline Modernization", "Observability Revamp"]),
    "Cybersecurity": dict(
        core=["Network Security", "Penetration Testing", "SIEM", "Splunk", "Vulnerability Assessment", "Incident Response",
              "Firewalls", "Wireshark", "Nmap", "Metasploit", "Threat Hunting", "Cryptography", "ISO 27001",
              "Malware Analysis", "IAM", "OWASP"],
        secondary=["Linux", "Python", "Burp Suite", "Zero Trust", "Digital Forensics", "SOC Operations", "Risk Assessment", "Security Auditing"],
        titles=["Security Analyst", "Cybersecurity Engineer", "Senior Security Engineer", "Lead Security Architect"],
        tasks=["security oversight programs", "vulnerability remediation cycles", "phishing response playbooks",
               "network segmentation projects", "red team engagements", "access governance reviews",
               "log correlation rules", "compliance readiness assessments", "endpoint hardening baselines", "threat intelligence feeds"],
        terms=["threat detection", "security operations", "risk mitigation", "compliance"],
        certs=["CompTIA Security+", "Certified Ethical Hacker", "CISSP", "GIAC Security Essentials"],
        degrees=[("B.Tech", "Computer Science"), ("B.Sc", "Cyber Security"), ("M.Tech", "Information Security"), ("B.E.", "Information Technology")],
        projects=["SOC Alert Triage Automation", "Internal Pentest Framework", "Perimeter Redesign Rollout", "Phishing Simulation Suite"]),
    "Human Resources": dict(
        core=["Recruitment", "Talent Acquisition", "Onboarding", "Employee Relations", "Payroll", "Performance Management",
              "HRIS", "Workday", "Compensation and Benefits", "Labor Law", "Training and Development",
              "Succession Planning", "Employer Branding", "Conflict Resolution", "HR Analytics", "Workforce Planning"],
        secondary=["ATS", "Excel", "Interviewing", "Engagement Surveys", "Exit Interviews", "Policy Drafting", "Campus Hiring", "Grievance Handling"],
        titles=["HR Executive", "HR Generalist", "Senior HR Business Partner", "HR Manager"],
        tasks=["end-to-end staffing drives", "new-joiner induction programs", "performance review cycles",
               "salary processing audits", "engagement survey initiatives", "leadership training calendars",
               "policy refresh projects", "campus outreach campaigns", "attrition reduction plans", "compensation benchmarking studies"],
        terms=["people operations", "talent management", "employee engagement", "HR compliance"],
        certs=["SHRM-CP", "PHR Certification", "Certified HR Analytics Professional", "CIPD Level 5"],
        degrees=[("MBA", "Human Resources"), ("B.Com", "Commerce"), ("BBA", "Business Administration"), ("M.A.", "Psychology")],
        projects=["Campus Hiring Drive", "Onboarding Experience Redesign", "Attrition Analytics Dashboard", "Competency Framework Rollout"]),
    "Digital Marketing": dict(
        core=["SEO", "SEM", "Google Analytics", "Google Ads", "Content Marketing", "Social Media Marketing",
              "Email Marketing", "Facebook Ads", "Marketing Automation", "HubSpot", "Copywriting",
              "Conversion Optimization", "Brand Strategy", "Campaign Management", "Influencer Marketing", "Keyword Research"],
        secondary=["A/B Testing", "Canva", "Excel", "Mailchimp", "Web Analytics", "CRM", "Landing Pages", "Marketing Analytics"],
        titles=["Marketing Executive", "Digital Marketing Specialist", "Senior Marketing Manager", "Head of Digital Marketing"],
        tasks=["paid acquisition campaigns", "organic traffic growth programs", "newsletter nurture journeys",
               "brand launch campaigns", "landing page experiments", "influencer collaborations",
               "social content calendars", "prospect acquisition funnels", "website conversion audits", "product launch promotions"],
        terms=["performance marketing", "digital campaigns", "audience growth", "brand visibility"],
        certs=["Google Ads Certification", "HubSpot Inbound Marketing", "Google Analytics Certification", "Meta Blueprint Certification"],
        degrees=[("MBA", "Marketing"), ("BBA", "Marketing"), ("B.A.", "Mass Communication"), ("B.Com", "Commerce")],
        projects=["Festive Sale Campaign", "Organic Search Revamp", "Email Lifecycle Program", "Creator Partnership Series"]),
    "Mechanical Engineering": dict(
        core=["AutoCAD", "SolidWorks", "CATIA", "ANSYS", "Finite Element Analysis", "CFD", "Thermodynamics",
              "Fluid Mechanics", "GD&T", "CNC Machining", "3D Printing", "HVAC", "Heat Transfer", "Mechanical Design",
              "Manufacturing Processes", "Product Development"],
        secondary=["Six Sigma", "Lean Manufacturing", "MATLAB", "PLC", "Quality Control", "Materials Science", "Project Management", "Root Cause Analysis"],
        titles=["Graduate Engineer Trainee", "Mechanical Engineer", "Senior Mechanical Engineer", "Lead Design Engineer"],
        tasks=["gearbox assembly designs", "thermal management systems", "tooling and fixture designs",
               "pressure vessel analyses", "production line improvements", "prototype development cycles",
               "pump performance studies", "component weight reduction programs", "ventilation layout designs", "failure analysis investigations"],
        terms=["product design", "manufacturing efficiency", "engineering analysis", "design validation"],
        certs=["Certified SolidWorks Professional", "Six Sigma Green Belt", "CATIA V5 Certification", "ASME GD&T Professional"],
        degrees=[("B.Tech", "Mechanical Engineering"), ("B.E.", "Mechanical Engineering"), ("M.Tech", "Thermal Engineering"), ("Diploma", "Mechanical Engineering")],
        projects=["Hydraulic Press Redesign", "Heat Exchanger Optimization", "Automated Fixture Design", "Turbine Blade Analysis"]),
    "Civil Engineering": dict(
        core=["AutoCAD", "Revit", "STAAD Pro", "Structural Analysis", "Surveying", "Construction Management",
              "Primavera", "Estimating", "Concrete Design", "Geotechnical Engineering", "Site Supervision",
              "BIM", "Quantity Surveying", "Hydrology", "Highway Design", "Building Codes"],
        secondary=["Project Management", "MS Project", "Soil Testing", "Civil 3D", "Safety Compliance", "Contract Administration", "Drafting", "Cost Control"],
        titles=["Site Engineer", "Civil Engineer", "Senior Structural Engineer", "Project Manager"],
        tasks=["residential tower structures", "highway widening works", "bridge rehabilitation packages",
               "drainage network designs", "on-site monitoring programs", "bill of quantities preparations",
               "foundation design reviews", "commercial complex constructions", "water supply schemes", "construction schedule baselines"],
        terms=["structural design", "site execution", "infrastructure projects", "construction quality"],
        certs=["PMP Certification", "Autodesk Revit Certified Professional", "OSHA Safety Certification", "LEED Green Associate"],
        degrees=[("B.Tech", "Civil Engineering"), ("B.E.", "Civil Engineering"), ("M.Tech", "Structural Engineering"), ("Diploma", "Civil Engineering")],
        projects=["Flyover Design Package", "Mid-rise Residential Block", "Stormwater Network Study", "Warehouse Foundation Design"]),
    "Finance & Accounting": dict(
        core=["Financial Modeling", "Financial Reporting", "Budgeting", "Forecasting", "Accounts Payable",
              "Accounts Receivable", "Taxation", "Auditing", "GAAP", "IFRS", "Tally", "SAP FICO", "QuickBooks",
              "Reconciliation", "Variance Analysis", "Cash Flow Management"],
        secondary=["Excel", "Risk Management", "Valuation", "Cost Accounting", "Payroll", "Bookkeeping", "Treasury", "Internal Controls"],
        titles=["Accounts Assistant", "Accountant", "Senior Financial Analyst", "Finance Manager"],
        tasks=["monthly closing cycles", "statutory audit support", "budget variance reviews",
               "vendor payment runs", "tax filing preparations", "cash flow forecasts",
               "intercompany reconciliations", "management reporting packs", "cost allocation models", "working capital improvements"],
        terms=["financial planning", "regulatory compliance", "financial analysis", "month-end reporting"],
        certs=["CPA", "CFA Level II", "ACCA Qualification", "Certified Management Accountant"],
        degrees=[("B.Com", "Accounting"), ("M.Com", "Finance"), ("MBA", "Finance"), ("B.Com", "Commerce")],
        projects=["Month-End Close Automation", "Budget Variance Dashboard", "Audit Readiness Program", "Cash Forecasting Model"]),
    "Sales & Business Development": dict(
        core=["Lead Generation", "Cold Calling", "CRM", "Salesforce", "Account Management", "Negotiation",
              "B2B Sales", "Pipeline Management", "Sales Forecasting", "Client Relationship Management",
              "Market Research", "Proposal Writing", "Territory Management", "Upselling", "Business Development", "Key Account Management"],
        secondary=["Presentation Skills", "Deal Closing", "Customer Retention", "HubSpot", "Excel", "Revenue Forecasting", "Channel Partnerships", "Contract Negotiation"],
        titles=["Sales Executive", "Sales Associate", "Senior Account Manager", "Regional Sales Manager"],
        tasks=["enterprise account portfolios", "new market entry plans", "outbound prospecting cadences",
               "channel partner programs", "annual sales plans", "renewal and upsell motions",
               "tender and proposal responses", "key account growth plans", "territory expansion drives", "pricing discussion cycles"],
        terms=["top-line expansion", "client acquisition", "customer relationships", "quota attainment"],
        certs=["Salesforce Certified Administrator", "HubSpot Sales Software Certification", "Certified Professional Sales Person", "Sandler Sales Training"],
        degrees=[("MBA", "Marketing"), ("BBA", "Business Administration"), ("B.Com", "Commerce"), ("B.A.", "Economics")],
        projects=["Regional Expansion Plan", "Enterprise Pipeline Rebuild", "Partner Channel Launch", "Renewal Revival Campaign"]),
}

# canonical skill -> extra aliases (the lower-cased canonical name is always an alias)
ALIASES = {
    "NLP": ["natural language processing"], "Machine Learning": ["ml"], "Deep Learning": ["dl", "neural networks"],
    "Scikit-learn": ["sklearn", "scikit learn"], "TensorFlow": ["tensor flow"], "PyTorch": ["pytorch"],
    "Pandas": [], "Data Visualization": ["data visualisation", "visualization"], "Power BI": ["powerbi", "power-bi"],
    "JavaScript": ["js", "java script"], "TypeScript": [], "Node.js": ["nodejs", "node js"],
    "REST APIs": ["rest api", "restful apis", "restful services"], "Microservices": ["micro-services", "microservice"],
    "PostgreSQL": ["postgres"], "MongoDB": ["mongo db", "mongo"], "Data Structures": ["dsa"],
    "CI/CD": ["ci cd", "continuous integration", "cicd"], "Kubernetes": ["k8s"], "Infrastructure as Code": ["iac"],
    "GCP": ["google cloud", "google cloud platform"], "AWS": ["amazon web services"], "Site Reliability": ["sre"],
    "Penetration Testing": ["pentesting", "pen testing"], "Vulnerability Assessment": ["vapt"], "SIEM": [],
    "Incident Response": ["incident handling"], "IAM": ["identity and access management"],
    "HRIS": ["hr information system"], "Talent Acquisition": ["talent sourcing"], "Recruitment": ["recruiting", "hiring"],
    "Compensation and Benefits": ["c&b", "compensation & benefits"], "Training and Development": ["learning and development"],
    "SEO": ["search engine optimization", "search engine optimisation"], "SEM": ["search engine marketing"],
    "Google Analytics": ["ga4"], "Social Media Marketing": ["smm"], "Email Marketing": ["email campaigns"],
    "Finite Element Analysis": ["fea"], "CFD": ["computational fluid dynamics"], "GD&T": ["geometric dimensioning and tolerancing"],
    "Revit": [], "BIM": ["building information modeling"], "Civil 3D": ["autodesk civil 3d"], "MS Project": ["microsoft project"],
    "GAAP": [], "IFRS": [], "SAP FICO": ["sap fi/co", "sap fico"], "QuickBooks": ["quick books"],
    "Financial Modeling": ["financial modelling"], "Accounts Payable": [], "Accounts Receivable": [],
    "CRM": ["customer relationship management"], "Salesforce": ["sfdc"], "B2B Sales": ["b2b sales"],
    "Key Account Management": ["kam"], "Client Relationship Management": ["client relations"],
    "SOC Operations": ["security operations center", "soc"], "ELK Stack": ["elasticsearch", "elk"],
    "A/B Testing": ["ab testing", "a/b tests", "split testing"], "Excel": ["ms excel", "microsoft excel", "advanced excel"],
    "Six Sigma": ["6 sigma"], "Lean Manufacturing": [], "Project Management": ["project planning"],
}

# Soft skills: used as realistic "background noise" in text and extracted separately.
SOFT_SKILLS = ["communication", "teamwork", "leadership", "problem solving", "time management",
               "stakeholder management", "adaptability", "critical thinking", "collaboration", "ownership"]

# Skill families (related technologies). Credit between family members is partial.
FAMILIES = [
    ["TensorFlow", "PyTorch", "Keras"], ["AWS", "Azure", "GCP"], ["Tableau", "Power BI"],
    ["Terraform", "Ansible", "CloudFormation"], ["Docker", "Kubernetes", "Helm"], ["Jenkins", "CI/CD", "GitOps"],
    ["Splunk", "SIEM", "ELK Stack"], ["Nmap", "Wireshark", "Metasploit", "Burp Suite"],
    ["SolidWorks", "CATIA", "AutoCAD"], ["ANSYS", "Finite Element Analysis", "CFD"],
    ["Revit", "BIM", "Civil 3D"], ["Primavera", "MS Project", "Project Management"],
    ["Google Ads", "SEM", "Facebook Ads"], ["SEO", "Keyword Research", "Content Marketing"],
    ["Salesforce", "HubSpot", "CRM"], ["Tally", "QuickBooks", "SAP FICO"], ["GAAP", "IFRS", "Financial Reporting"],
    ["Recruitment", "Talent Acquisition", "Interviewing"], ["HRIS", "Workday"], ["React", "Angular", "TypeScript"],
    ["PostgreSQL", "SQL", "MongoDB"], ["Spark", "Pandas"], ["Penetration Testing", "Vulnerability Assessment", "OWASP"],
    ["Budgeting", "Forecasting", "Variance Analysis"],
]
# child -> parent: knowing the child implies working knowledge of the parent.
IMPLIES = [
    ("Pandas", "Python"), ("NumPy", "Python"), ("Scikit-learn", "Python"), ("Keras", "TensorFlow"),
    ("Spring Boot", "Java"), ("React", "JavaScript"), ("Node.js", "JavaScript"), ("Angular", "TypeScript"),
    ("XGBoost", "Machine Learning"), ("Deep Learning", "Machine Learning"), ("Helm", "Kubernetes"),
    ("Splunk", "SIEM"), ("Metasploit", "Penetration Testing"), ("Burp Suite", "Penetration Testing"),
    ("STAAD Pro", "Structural Analysis"), ("Google Ads", "SEM"), ("Workday", "HRIS"), ("SAP FICO", "Financial Reporting"),
    ("Primavera", "Project Management"), ("Salesforce", "CRM"), ("Mailchimp", "Email Marketing"), ("Civil 3D", "AutoCAD"),
]
# Relations the extractor does NOT know about (used only for ground-truth relevance).
HELD_OUT_FAMILIES = {i for i in range(len(FAMILIES)) if i % 4 == 3}
HELD_OUT_IMPLIES = {i for i in range(len(IMPLIES)) if i % 4 == 2}

# ---- Evidence verbs ---------------------------------------------------------------------
# Generator vocabulary (rich). Level 1 = exposure ... Level 4 = ownership/leadership.
GEN_VERBS = {
    1: ["Familiar with {s}", "Exposure to {s} through coursework", "Basic knowledge of {s}", "Assisted the team with {s} tasks",
        "Introduced to {s} during training", "Learning {s} through online courses", "Shadowed senior colleagues working with {s}",
        "Helped with {s} related activities", "Aided colleagues using {s}"],
    2: ["Used {s} to support {t}", "Worked with {s} for {t}", "Supported {t} using {s}", "Contributed to {t} with {s}",
        "Maintained {t} in {s}", "Utilized {s} for {t}", "Participated in {t} leveraging {s}", "Handled {t} using {s}"],
    3: ["Developed {t} using {s}", "Implemented {t} with {s}", "Designed and built {t} in {s}", "Delivered {t} using {s}",
        "Automated {t} with {s}", "Optimized {t} through {s}", "Engineered {t} leveraging {s}", "Crafted {t} using {s}",
        "Established {t} based on {s}", "Executed {t} with {s}"],
    4: ["Led {t} using {s}", "Architected {t} on {s}", "Owned end-to-end {t} using {s}", "Spearheaded {t} built on {s}",
        "Mentored a team of {n} on {s} while driving {t}", "Drove strategy for {t} with {s}", "Pioneered {t} using {s}",
        "Championed {t} powered by {s}", "Orchestrated {t} across teams with {s}"],
}
# Extractor vocabulary (strictly narrower than the generator's; unseen verbs fall back to level 2).
EXT_VERB_PATTERNS = {
    1: r"\b(familiar with|exposure to|basic knowledge|assisted|introduced to|learning|shadow\w*|aided)\b",
    2: r"\b(used|worked with|supported|contributed|maintained|utili[sz]ed|participated|handled)\b",
    3: r"\b(developed|implemented|designed|built|delivered|automated|optimi[sz]ed|engineered|created|deployed|configured)\b",
    4: r"\b(led|architect\w*|owned|spearhead\w*|mentored|drove|managed a team|directed)\b",
}
METRIC_PATTERN = (r"(\b\d{1,3}(?:\.\d+)?\s?%|\b\d+(?:\.\d+)?\s?[xX]\b|\$\s?\d+(?:\.\d+)?\s?[kKmM]?|"
                  r"\b\d+(?:\.\d+)?\s?[kKmM]\+?\s+(?:users|requests|records|clients|customers|accounts|leads|visitors)\b|"
                  r"\b\d{2,}\s+(?:users|clients|customers|accounts|leads|employees|vendors)\b)")

SECTION_HEADERS = {
    "summary": ["summary", "professional summary", "profile", "objective", "career objective", "about me"],
    "experience": ["experience", "work experience", "professional experience", "work history", "employment history", "career history"],
    "education": ["education", "academic background", "educational qualifications", "academics"],
    "skills": ["skills", "technical skills", "core competencies", "key skills", "skills & tools", "areas of expertise", "competencies"],
    "projects": ["projects", "selected projects", "academic projects", "key projects", "project experience"],
    "certifications": ["certifications", "licenses & certifications", "courses and certifications", "certificates"],
    "activities": ["activities", "extracurricular activities", "interests", "affiliations", "volunteering", "interests & activities"],
}

TITLE_ROLE_WORDS = (r"(engineer|analyst|scientist|manager|developer|executive|officer|accountant|recruiter|specialist|"
                    r"architect|consultant|administrator|associate|trainee|assistant|generalist|partner|head|lead|director|supervisor)")
