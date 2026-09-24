"""
SkillGraph & Student Growth Intelligence - Data Engine & Backend API
Analyzes student performance data, computes persona archetypes, detects intervention alerts,
and powers the REST API for both the Mentor Console and Student Portal.
"""

import math
import os
from pathlib import Path
from typing import List, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse

# Resolve the directory where hackathon.py lives
BASE_DIR = Path(__file__).resolve().parent

# Initialize FastAPI App
app = FastAPI(
    title="Student Growth Intelligence & SkillGraph API",
    description="Backend API powering persona detection, risk alerts, team matching, and telemetry for mentors & students.",
    version="2.0.0"
)

# Enable CORS for local web dev & frontend portals
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# 1. CORE DATA LOGIC & PERSONA DETECTOR
# -----------------------------------------------------------------------------

def load_and_enrich_data(csv_path: str = None) -> pd.DataFrame:
    if csv_path is None:
        csv_path = str(BASE_DIR / "students.csv")
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        df = pd.read_csv(str(BASE_DIR / "students.csv"))

    def compute_persona(r):
        """
        Determines the dominant student archetype using normalized 0-100 scores:
        - Competitor: High DSA/coding challenge velocity (LeetCode + HackerRank)
        - Builder: Hands-on project maker & hackathon winner (GitHub + wins)
        - Leader: Soft skills, ideathons, club/community leadership
        - Scholar: Academic excellence & research (CGPA + paper presentations)
        """
        lc   = float(r.get('leetcode_solved', 0))
        hr   = float(r.get('hackerrank_solved', 0))
        gp   = float(r.get('github_projects', 0))
        hw   = float(r.get('hackathons_won', 0))
        lead = float(r.get('leadership', 0))
        iw   = float(r.get('ideathons_won', 0))
        cgpa = float(r.get('cgpa', 0))
        pp   = float(r.get('paper_presentations', 0))

        # Normalize each dimension to 0-100
        competitor = min(100.0, (lc / 350.0) * 60 + (hr / 250.0) * 40)
        builder    = min(100.0, (gp / 7.0)   * 55 + (hw / 2.0)   * 45)
        leader     = min(100.0, (lead * 55)       + (iw / 2.0)   * 45)
        scholar    = min(100.0, (max(0, cgpa - 7.0) / 2.8) * 50 + (pp / 2.0) * 50)

        scores = {
            "Competitor": competitor,
            "Builder":    builder,
            "Leader":     leader,
            "Scholar":    scholar,
        }
        return max(scores, key=scores.get)

    def compute_alerts(r):
        """Identifies proactive intervention flags for mentors."""
        alerts_list = []
        cgpa = float(r.get('cgpa', 0))
        leetcode = int(r.get('leetcode_solved', 0))
        year = int(r.get('year', 1))
        internships = int(r.get('internships', 0))
        hackathons = int(r.get('hackathons_participated', 0))

        if cgpa < 7.5:
            alerts_list.append({
                "severity": "HIGH",
                "code": "CGPA_DROP",
                "title": "Low Academic Standing",
                "reason": f"CGPA is {cgpa:.2f}, below department target (7.50)"
            })
        if leetcode < 50 and year >= 2:
            alerts_list.append({
                "severity": "MEDIUM",
                "code": "CODING_VELOCITY",
                "title": "Low DSA Velocity",
                "reason": f"Only {leetcode} LeetCode problems solved by Year {year}"
            })
        if internships == 0 and year >= 3:
            alerts_list.append({
                "severity": "HIGH",
                "code": "CAREER_READINESS",
                "title": "No Internship Experience",
                "reason": f"Year {year} student has 0 recorded corporate internships"
            })
        if hackathons == 0 and year >= 2:
            alerts_list.append({
                "severity": "LOW",
                "code": "EVENT_ENGAGEMENT",
                "title": "Zero Hackathon Participation",
                "reason": "Has not participated in any competitive hackathons"
            })
        return alerts_list

    def compute_growth_score(r):
        """Calculates a normalized 0-100 holistic growth index."""
        cgpa_score = min(100.0, (float(r.get('cgpa', 0)) / 10.0) * 100)
        coding_score = min(100.0, (float(r.get('leetcode_solved', 0)) / 400.0) * 100)
        project_score = min(100.0, (float(r.get('github_projects', 0)) / 8.0) * 100)
        events_score = min(100.0, (float(r.get('hackathons_participated', 0)) / 6.0) * 100)
        exp_score = min(100.0, (float(r.get('internships', 0)) / 2.0) * 100)

        # Weighted composite: 30% Academics, 25% Coding, 20% Projects, 15% Events, 10% Exp
        index = (cgpa_score * 0.30) + (coding_score * 0.25) + (project_score * 0.20) + (events_score * 0.15) + (exp_score * 0.10)
        return round(index, 1)

    df['persona'] = df.apply(compute_persona, axis=1)
    df['growth_index'] = df.apply(compute_growth_score, axis=1)
    df['alerts'] = df.apply(compute_alerts, axis=1)

    return df

# Load enriched dataframe
df_students = load_and_enrich_data()

# Save enriched CSV relative to this file's directory
enriched_path = str(BASE_DIR / "students_enriched.csv")
df_students.to_csv(enriched_path, index=False)




# -----------------------------------------------------------------------------
# 2. REST API ENDPOINTS
# -----------------------------------------------------------------------------

@app.get("/")
def root():
    """Serve the Mentor Console (index.html) as the root page."""
    return FileResponse(str(BASE_DIR / "index.html"))

@app.get("/api/status")
def api_status():
    """Health-check endpoint — returns JSON status."""
    return {
        "status": "online",
        "service": "SkillGraph & Student Growth Intelligence",
        "total_students": len(df_students),
        "docs_url": "/docs"
    }

# Catch-all: serve any .html page directly from the project root
@app.get("/{page_name}.html")
def serve_page(page_name: str):
    """Serve any top-level HTML page (profile.html, skills.html, etc.)."""
    page_path = BASE_DIR / f"{page_name}.html"
    if page_path.exists():
        return FileResponse(str(page_path))
    raise HTTPException(status_code=404, detail=f"Page '{page_name}.html' not found.")

@app.get("/api/stats")
def get_stats():
    """Returns college and cohort-wide telemetry for mentor dashboard."""
    flagged_students = sum(1 for _, r in df_students.iterrows() if len(r['alerts']) > 0)
    persona_counts = df_students['persona'].value_counts().to_dict()

    year_stats = {}
    for y in sorted(df_students['year'].unique()):
        sub = df_students[df_students['year'] == y]
        year_stats[f"Year {y}"] = {
            "students": len(sub),
            "avg_cgpa": round(float(sub['cgpa'].mean()), 2),
            "avg_growth": round(float(sub['growth_index'].mean()), 1),
            "total_hackathons": int(sub['hackathons_participated'].sum()),
            "total_internships": int(sub['internships'].sum()),
            "flagged": sum(1 for _, r in sub.iterrows() if len(r['alerts']) > 0)
        }

    return {
        "total_students": len(df_students),
        "avg_cgpa": round(float(df_students['cgpa'].mean()), 2),
        "avg_growth_index": round(float(df_students['growth_index'].mean()), 1),
        "total_hackathons_participated": int(df_students['hackathons_participated'].sum()),
        "total_hackathons_won": int(df_students['hackathons_won'].sum()),
        "total_internships": int(df_students['internships'].sum()),
        "zero_internship_count": int((df_students['internships'] == 0).sum()),
        "flagged_count": flagged_students,
        "persona_breakdown": persona_counts,
        "cohorts": year_stats
    }

@app.get("/api/students")
def get_students(year: Optional[int] = None, persona: Optional[str] = None, flagged_only: bool = False):
    """Fetches students with optional filtering by year, persona archetype, or flagged state."""
    df_filtered = df_students.copy()
    if year is not None:
        df_filtered = df_filtered[df_filtered['year'] == year]
    if persona:
        df_filtered = df_filtered[df_filtered['persona'].str.lower() == persona.lower()]
    if flagged_only:
        df_filtered = df_filtered[df_filtered['alerts'].apply(lambda x: len(x) > 0)]

    results = []
    for _, r in df_filtered.iterrows():
        item = r.to_dict()
        results.append(item)
    return results

@app.get("/api/student/{student_id}")
def get_student_detail(student_id: str):
    """Fetches comprehensive profile details and skill metrics for a specific student."""
    match = df_students[df_students['student_id'].str.upper() == student_id.upper()]
    if match.empty:
        # Also check name search if ID not matched
        match = df_students[df_students['name'].str.lower().str.contains(student_id.lower())]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Student '{student_id}' not found.")

    student = match.iloc[0].to_dict()
    
    # Calculate radar chart competencies (0-100)
    radar = {
        "academics": min(100, int(student['cgpa'] * 10)),
        "programming": min(100, int((student['leetcode_solved'] / 400.0) * 100)),
        "projects": min(100, int((student['github_projects'] / 8.0) * 100)),
        "events": min(100, int((student['hackathons_participated'] / 6.0) * 100)),
        "leadership": 85 if student['leadership'] == 1 else 45
    }

    # Generate tailored skill tags based on department & persona
    tech_skills = []
    if student['department'] in ['Computer Science', 'Information Technology']:
        tech_skills = ["Python", "SQL", "Data Structures", "Web Development", "FastAPI"]
    elif student['department'] == 'Electronics':
        tech_skills = ["Embedded C", "IoT", "Python", "Verilog", "Signal Processing"]
    elif student['department'] == 'Mechanical':
        tech_skills = ["AutoCAD", "SolidWorks", "Python for CAD", "ANSYS", "Robotics"]
    else:
        tech_skills = ["MATLAB", "PLC/SCADA", "Python", "Circuit Design", "Power Systems"]

    return {
        "profile": student,
        "radar": radar,
        "skills": tech_skills,
        "persona": student['persona'],
        "growth_index": student['growth_index'],
        "alerts": student['alerts']
    }

@app.get("/api/team-match/{student_id}")
def get_team_matches(student_id: str):
    """
    Intelligent Teammate Recommendation Algorithm:
    Finds peers with complementary archetypes and skillsets to build balanced hackathon squads.
    E.g. A Competitor needs a Builder and a Leader; a Scholar needs a Competitor and Builder.
    """
    match = df_students[df_students['student_id'].str.upper() == student_id.upper()]
    if match.empty:
        student_obj = df_students.iloc[0]
    else:
        student_obj = match.iloc[0]

    current_persona = student_obj['persona']
    current_dept = student_obj['department']

    # Target complementary personas
    compliment_map = {
        "Competitor": ["Builder", "Leader", "Scholar"],
        "Builder": ["Competitor", "Leader", "Scholar"],
        "Leader": ["Competitor", "Builder", "Scholar"],
        "Scholar": ["Builder", "Competitor", "Leader"]
    }
    targets = compliment_map.get(current_persona, ["Builder", "Competitor"])

    candidates = df_students[
        (df_students['student_id'] != student_obj['student_id']) &
        (df_students['persona'].isin(targets))
    ].copy()

    # Calculate match affinity score
    def match_score(r):
        score = 70.0
        # Diversity in persona
        if r['persona'] != current_persona:
            score += 15.0
        # Interdisciplinary boost
        if r['department'] != current_dept:
            score += 8.0
        # High growth bonus
        score += min(10.0, (r['growth_index'] / 10.0))
        # Ensure within 75% to 98%
        return min(98, int(score))

    candidates['match_percentage'] = candidates.apply(match_score, axis=1)
    candidates = candidates.sort_values(by='match_percentage', ascending=False).head(4)

    results = []
    for _, c in candidates.iterrows():
        results.append({
            "student_id": c['student_id'],
            "name": c['name'],
            "department": c['department'],
            "year": int(c['year']),
            "persona": c['persona'],
            "match_percentage": int(c['match_percentage']),
            "cgpa": float(c['cgpa']),
            "leetcode_solved": int(c['leetcode_solved']),
            "github_projects": int(c['github_projects']),
            "hackathons_won": int(c['hackathons_won'])
        })

    return {
        "target_student": student_obj['name'],
        "target_persona": current_persona,
        "recommendations": results
    }

@app.get("/api/students/search")
def search_students(q: str = ""):
    """Fast search across students by ID, Name, Department, or Persona."""
    q_clean = q.strip().lower()
    if not q_clean:
        return df_students.head(20).to_dict(orient="records")
    
    mask = (
        df_students['student_id'].str.lower().str.contains(q_clean) |
        df_students['name'].str.lower().str.contains(q_clean) |
        df_students['department'].str.lower().str.contains(q_clean) |
        df_students['persona'].str.lower().str.contains(q_clean)
    )
    return df_students[mask].to_dict(orient="records")

@app.get("/api/learning-path/{student_id}")
def get_learning_path(student_id: str):
    """
    Intelligent Adaptive Learning Engine:
    Constructs high-impact milestone steps customized to the student's active risk alerts,
    current year, department, and persona archetype.
    """
    match = df_students[df_students['student_id'].str.upper() == student_id.upper()]
    if match.empty:
        match = df_students[df_students['name'].str.lower().str.contains(student_id.lower())]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Student '{student_id}' not found.")
    
    student = match.iloc[0].to_dict()
    persona = student['persona']
    alerts = student['alerts']
    year = int(student['year'])
    dept = student['department']

    steps = []
    step_num = 1

    # 1. Address High/Medium Priority Alerts First (Intervention)
    for alert in alerts:
        if alert['code'] == 'CGPA_DROP':
            steps.append({
                "step": step_num,
                "title": "Academic Recovery & GPA Fortification",
                "category": "Academics",
                "urgency": "High",
                "description": f"Meet with faculty mentor to review core subjects. Aim to lift current {student['cgpa']:.2f} CGPA above the 7.50 target before final exams.",
                "action": "Schedule 1-on-1 Faculty Review",
                "completed": False
            })
            step_num += 1
        elif alert['code'] == 'CODING_VELOCITY':
            steps.append({
                "step": step_num,
                "title": "DSA Acceleration Sprint",
                "category": "Programming",
                "urgency": "High",
                "description": f"Complete the 'Blind 75' / NeetCode DSA curriculum. Target 5 problems/week to cross the 150+ milestone on LeetCode.",
                "action": "Start NeetCode 75 Roadmap",
                "completed": False
            })
            step_num += 1
        elif alert['code'] == 'CAREER_READINESS':
            steps.append({
                "step": step_num,
                "title": "Pre-Placement Internship Fast-Track",
                "category": "Career",
                "urgency": "High",
                "description": "Zero industry internships on record. Refine resume, prepare portfolio, and apply for verified summer/winter internships via college placement cell.",
                "action": "Submit 10 Verified Internship Applications",
                "completed": False
            })
            step_num += 1
        elif alert['code'] == 'EVENT_ENGAGEMENT':
            steps.append({
                "step": step_num,
                "title": "Competitive Hackathon Debut",
                "category": "Hackathons",
                "urgency": "Medium",
                "description": "Form a team using the Team Match feature and register for an upcoming inter-college or national hackathon (Smart India Hackathon, Devfolio).",
                "action": "Register Hackathon Squad",
                "completed": False
            })
            step_num += 1

    # 2. Archetype-Tailored Milestones
    if persona == 'Competitor':
        steps.append({
            "step": step_num,
            "title": "Advanced Graph & Dynamic Programming Mastery",
            "category": "DSA",
            "urgency": "Normal",
            "description": "Solve 40 hard-tier LeetCode problems focusing on Dijkstra, Segment Trees, and Multi-Dimensional DP.",
            "action": "Open LeetCode Study Plan",
            "completed": True if student['leetcode_solved'] > 250 else False
        })
        step_num += 1
        steps.append({
            "step": step_num,
            "title": "Weekly Rated Contest Ranking",
            "category": "Contests",
            "urgency": "Normal",
            "description": "Compete in consecutive Codeforces / LeetCode global contests to maintain 1800+ rating percentile.",
            "action": "Register for Next Contest",
            "completed": False
        })
        step_num += 1
        steps.append({
            "step": step_num,
            "title": "System Design & Low-Level Design (LLD)",
            "category": "Engineering",
            "urgency": "Normal",
            "description": "Design an in-memory Key-Value store and rate limiter with concurrency locks in Python/Java.",
            "action": "View Design Blueprint",
            "completed": False
        })
        step_num += 1

    elif persona == 'Builder':
        steps.append({
            "step": step_num,
            "title": "Production Full-Stack Cloud Deployment",
            "category": "DevOps",
            "urgency": "Normal",
            "description": "Containerize your flagship web application with Docker and deploy to AWS / GCP with automated CI/CD pipeline.",
            "action": "Setup GitHub Actions CI/CD",
            "completed": True if student['github_projects'] >= 4 else False
        })
        step_num += 1
        steps.append({
            "step": step_num,
            "title": "Open Source Contributions",
            "category": "Open Source",
            "urgency": "Normal",
            "description": "Submit 2 meaningful pull requests to popular open-source frameworks (FastAPI, LangChain, or React ecosystems).",
            "action": "Explore Good First Issues",
            "completed": False
        })
        step_num += 1
        steps.append({
            "step": step_num,
            "title": "Architect Microservices System",
            "category": "Architecture",
            "urgency": "Normal",
            "description": "Implement asynchronous message queues (RabbitMQ / Kafka) to connect multiple microservices.",
            "action": "Read System Specs",
            "completed": False
        })
        step_num += 1

    elif persona == 'Leader':
        steps.append({
            "step": step_num,
            "title": "Hackathon Squad Captaincy",
            "category": "Leadership",
            "urgency": "Normal",
            "description": "Assemble a 4-person diverse squad (Builder + Competitor + Scholar) and lead project delivery for upcoming hackathon.",
            "action": "Open Team Matcher",
            "completed": True if student['hackathons_won'] > 0 else False
        })
        step_num += 1
        steps.append({
            "step": step_num,
            "title": "Lead College Technical Workshop",
            "category": "Community",
            "urgency": "Normal",
            "description": "Conduct a hands-on technical workshop on Modern Web or AI for 50+ junior students in your department.",
            "action": "Submit Workshop Proposal",
            "completed": False
        })
        step_num += 1
        steps.append({
            "step": step_num,
            "title": "Product Pitch & Ideation Deck",
            "category": "Product",
            "urgency": "Normal",
            "description": "Create a 10-slide startup pitch deck with market sizing and UI prototypes for an inter-college Ideathon.",
            "action": "Download Pitch Template",
            "completed": False
        })
        step_num += 1

    else:  # Scholar
        steps.append({
            "step": step_num,
            "title": "IEEE / Scopus Research Manuscript Draft",
            "category": "Research",
            "urgency": "Normal",
            "description": "Complete experimental benchmark analysis and co-author a conference paper with department faculty.",
            "action": "Collaborate in Overleaf",
            "completed": True if student['paper_presentations'] > 0 else False
        })
        step_num += 1
        steps.append({
            "step": step_num,
            "title": "Advanced Algorithmic Foundations",
            "category": "Theory",
            "urgency": "Normal",
            "description": "Deep-dive into probabilistic algorithms and computational complexity theory for academic honors.",
            "action": "Access Lecture Notes",
            "completed": False
        })
        step_num += 1
        steps.append({
            "step": step_num,
            "title": "Maintain Dean's Honor Roll Standing",
            "category": "Academics",
            "urgency": "Normal",
            "description": "Sustain academic excellence with semester GPA > 9.0 to qualify for research fellowships.",
            "action": "Track Course Credits",
            "completed": True if student['cgpa'] >= 9.0 else False
        })
        step_num += 1

    return {
        "student_id": student['student_id'],
        "name": student['name'],
        "persona": persona,
        "department": dept,
        "year": year,
        "total_steps": len(steps),
        "completed_count": sum(1 for s in steps if s['completed']),
        "steps": steps
    }

@app.get("/api/student/{student_id}/portfolio")
def get_student_portfolio(student_id: str):
    """Returns rich personalized project, certification, and achievement portfolio items."""
    match = df_students[df_students['student_id'].str.upper() == student_id.upper()]
    if match.empty:
        match = df_students[df_students['name'].str.lower().str.contains(student_id.lower())]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Student '{student_id}' not found.")
    
    s = match.iloc[0].to_dict()
    dept = s['department']
    persona = s['persona']
    num_proj = int(s['github_projects'])
    num_certs = int(s.get('certifications_count', 3))
    hack_won = int(s['hackathons_won'])
    hack_part = int(s['hackathons_participated'])
    idea_won = int(s['ideathons_won'])
    papers = int(s['paper_presentations'])

    all_projects = [
        {
            "id": "proj-1",
            "title": "SkillGraph & Growth Intelligence Engine",
            "category": "Web Application & AI",
            "icon": "📊",
            "desc": "Full-stack institutional growth tracker with automated persona detection and ML radar graphs.",
            "tech": ["FastAPI", "Python", "Chart.js", "Vanilla CSS"],
            "repo": "https://github.com/student/skillgraph",
            "stars": 42,
            "featured": True
        },
        {
            "id": "proj-2",
            "title": "Industrial Satellite Fire Detection",
            "category": "AI / Computer Vision",
            "icon": "🔥",
            "desc": "Real-time thermal anomaly identification using multi-spectral satellite imagery and CNN models.",
            "tech": ["PyTorch", "OpenCV", "Flask", "Docker"],
            "repo": "https://github.com/student/fire-detect-sat",
            "stars": 28,
            "featured": True
        },
        {
            "id": "proj-3",
            "title": "Campus 360 Virtual Interactive Tour",
            "category": "Interactive Web",
            "icon": "🎓",
            "desc": "Panoramic 3D web experience allowing prospective students to explore college labs and classrooms.",
            "tech": ["Three.js", "WebGL", "JavaScript"],
            "repo": "https://github.com/student/campus360",
            "stars": 19,
            "featured": False
        },
        {
            "id": "proj-4",
            "title": "Distributed High-Concurrency Cache",
            "category": "Systems & Backend",
            "icon": "⚡",
            "desc": "In-memory LRU key-value cache with raft consensus replication and RESTful health monitoring.",
            "tech": ["Go", "gRPC", "Redis", "Docker"],
            "repo": "https://github.com/student/distributed-cache",
            "stars": 35,
            "featured": False
        },
        {
            "id": "proj-5",
            "title": "IoT Smart Energy & Grid Monitor",
            "category": "IoT & Embedded",
            "icon": "⚡",
            "desc": "Edge micro-controller telemetry streaming sensor data to cloud dashboard with power spike alerts.",
            "tech": ["ESP32", "MQTT", "Python", "Grafana"],
            "repo": "https://github.com/student/iot-grid-telemetry",
            "stars": 15,
            "featured": False
        },
        {
            "id": "proj-6",
            "title": "Automated Code Assessment Sandbox",
            "category": "Dev Tools",
            "icon": "🧪",
            "desc": "Isolated Linux container runner for running unit tests against student DSA problem submissions.",
            "tech": ["Docker SDK", "Python", "PostgreSQL"],
            "repo": "https://github.com/student/code-sandbox",
            "stars": 51,
            "featured": False
        }
    ]
    user_projects = all_projects[:max(3, min(len(all_projects), num_proj))]

    all_certs = [
        {
            "title": "AWS Certified Cloud Practitioner",
            "issuer": "Amazon Web Services",
            "icon": "☁️",
            "date": "Jan 2024",
            "cred_id": "AWS-CCP-92841",
            "status": "Verified",
            "skills": "Cloud Computing, EC2, S3, IAM"
        },
        {
            "title": "Python for Data Science & ML",
            "issuer": "Coursera / DeepLearning.AI",
            "icon": "🐍",
            "date": "Aug 2023",
            "cred_id": "COUR-PY-51209",
            "status": "Verified",
            "skills": "Python, Pandas, NumPy, Scikit-Learn"
        },
        {
            "title": "Meta Professional Front-End Developer",
            "issuer": "Meta",
            "icon": "💻",
            "date": "Nov 2023",
            "cred_id": "META-FED-88412",
            "status": "Verified",
            "skills": "JavaScript, HTML5, CSS3, React"
        },
        {
            "title": "PostgreSQL & Database Design",
            "issuer": "freeCodeCamp",
            "icon": "🗄️",
            "date": "Mar 2024",
            "cred_id": "FCC-SQL-10398",
            "status": "Verified",
            "skills": "Relational DB, SQL Queries, Indexing"
        },
        {
            "title": "Docker Essentials & Containers",
            "issuer": "Linux Foundation",
            "icon": "🐳",
            "date": "May 2024",
            "cred_id": "LF-DKR-30194",
            "status": "Verified",
            "skills": "Docker, Containers, CI/CD"
        }
    ]
    user_certs = all_certs[:max(2, min(len(all_certs), num_certs))]

    user_achievements = []
    if hack_won > 0:
        user_achievements.append({
            "title": f"Inter-College Hackathon Winner ({hack_won}x First Place)",
            "subtitle": "National Smart Hackathon / DevHack",
            "icon": "🥇",
            "badge": "1st Prize",
            "year": "2024",
            "desc": f"Secured top position out of 80+ collegiate teams for building an AI-powered growth intelligence platform."
        })
    if hack_part > 0 and hack_won == 0:
        user_achievements.append({
            "title": f"Hackathon Finalist ({hack_part} Participations)",
            "subtitle": "Zonal Hackathon Summit",
            "icon": "🏅",
            "badge": "Finalist",
            "year": "2023",
            "desc": "Built collaborative prototype within 24-hour sprint and presented to industry judges."
        })
    if idea_won > 0:
        user_achievements.append({
            "title": f"Ideathon Innovation Champion ({idea_won}x)",
            "subtitle": "Institutional Innovation Council",
            "icon": "💡",
            "badge": "Champion",
            "year": "2024",
            "desc": "Recognized for high-impact sustainable tech startup concept and pitch."
        })
    if papers > 0:
        user_achievements.append({
            "title": f"Research Publication Presentation ({papers}x)",
            "subtitle": "IEEE International Tech Conference",
            "icon": "📜",
            "badge": "Published",
            "year": "2024",
            "desc": "Authored and presented peer-reviewed technical paper on intelligent student telemetry."
        })
    if s.get('leadership', 0) == 1:
        user_achievements.append({
            "title": "Student Technical Club Lead & Coordinator",
            "subtitle": "College Coding & Innovation Club",
            "icon": "🎤",
            "badge": "Lead",
            "year": "2023 - Present",
            "desc": "Spearheaded peer programming bootcamps and mentoring circles for 120+ student members."
        })
    if float(s['cgpa']) >= 8.5:
        user_achievements.append({
            "title": "Dean's Academic Excellence Honor Roll",
            "subtitle": f"Top 10% in {dept}",
            "icon": "🎓",
            "badge": f"CGPA {s['cgpa']}",
            "year": "2023 - 2024",
            "desc": f"Consistently maintained outstanding academic performance with a {s['cgpa']} cumulative GPA."
        })

    return {
        "student_id": s['student_id'],
        "name": s['name'],
        "projects": user_projects,
        "certifications": user_certs,
        "achievements": user_achievements,
        "metrics": {
            "total_projects": num_proj,
            "total_certifications": num_certs,
            "hackathons_won": hack_won,
            "hackathons_participated": hack_part,
            "paper_presentations": papers,
            "internships": int(s['internships'])
        }
    }

# -----------------------------------------------------------------------------
# 3. CLI DIAGNOSTIC RUNNER
# -----------------------------------------------------------------------------

def print_cli_summary():
    print("=" * 60)
    print("  STUDENT GROWTH INTELLIGENCE & SKILLGRAPH CONSOLE")
    print("=" * 60)
    print(f"Total students loaded & enriched: {len(df_students)}")
    print(f"Average CGPA:                     {df_students['cgpa'].mean():.2f}")
    print(f"Total Hackathons Participated:    {df_students['hackathons_participated'].sum()}")
    print(f"Total Hackathons Won:             {df_students['hackathons_won'].sum()}")
    print(f"Total Internships:                {df_students['internships'].sum()}")
    print(f"Students with 0 Internships:      {(df_students['internships'] == 0).sum()}")
    print()

    print(">>> PERSONA DISTRIBUTION:")
    for persona, count in df_students['persona'].value_counts().items():
        bar = "#" * (count // 2)
        print(f"  {persona:<12}: {count:>3} students  {bar}")
    print()

    flagged = df_students[df_students['alerts'].apply(lambda x: len(x) > 0)]
    print(f">>> MENTOR INTERVENTION ALERTS ({len(flagged)} students flagged):")
    for _, s in flagged.head(6).iterrows():
        top_alert = s['alerts'][0]
        print(f"  [{top_alert['severity']}] {s['student_id']} - {s['name']:<18} ({s['department']}, Yr {s['year']}) : {top_alert['title']}")
    print()
    print(">>> Enriched dataset saved to: students_enriched.csv")
    print(">>> To launch API server: uvicorn hackathon:app --reload --port 8000")
    print("=" * 60)

if __name__ == "__main__":
    print_cli_summary()

# ===================================================================
# STATIC FILE SERVING — must be LAST so all /api/* routes take priority
# ===================================================================
# Serves style.css, script.js, and any other file in the project folder
# at their natural paths: http://localhost:8000/style.css, etc.
app.mount("/", StaticFiles(directory=str(BASE_DIR), html=True), name="frontend")
