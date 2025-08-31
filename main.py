import streamlit as st
import os
import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
import requests
from bs4 import BeautifulSoup
import random
import google.generativeai as genai

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic Models
class CourseDetail(BaseModel):
    course_name: str = Field(default="")
    platform: str = Field(default="")
    link: str = Field(default="")
    duration: str = Field(default="")
    difficulty_level: str = Field(default="")
    prerequisites: List[str] = Field(default_factory=list)
    key_topics: List[str] = Field(default_factory=list)
    certification: bool = Field(default=False)
    price: Optional[str] = Field(default=None)

class ProjectDetail(BaseModel):
    title: str = Field(default="")
    description: str = Field(default="")
    skills_practiced: List[str] = Field(default_factory=list)
    difficulty_level: str = Field(default="")
    estimated_duration: str = Field(default="")
    resources_needed: List[str] = Field(default_factory=list)
    learning_outcomes: List[str] = Field(default_factory=list)
    implementation_steps: List[str] = Field(default_factory=list)

class SkillDetail(BaseModel):
    skill_name: str = Field(default="")
    importance_level: str = Field(default="")
    time_to_master: str = Field(default="")
    prerequisites: List[str] = Field(default_factory=list)
    resources: List[str] = Field(default_factory=list)
    industry_applications: List[str] = Field(default_factory=list)
    proficiency_metrics: List[str] = Field(default_factory=list)

class CareerPath(BaseModel):
    title: str = Field(default="")
    description: str = Field(default="")
    salary_range: str = Field(default="")
    required_experience: str = Field(default="")
    key_responsibilities: List[str] = Field(default_factory=list)
    required_skills: List[str] = Field(default_factory=list)
    growth_opportunities: List[str] = Field(default_factory=list)
    industry_demand: str = Field(default="")
    typical_job_titles: List[str] = Field(default_factory=list)

class JobListing(BaseModel):
    job_title: str = Field(default="")
    company: str = Field(default="")
    location: str = Field(default="")
    platform: str = Field(default="")
    link: str = Field(default="")
    posted_date: str = Field(default="")
    description: str = Field(default="")

class CareerGuidance(BaseModel):
    domain_overview: str = Field(default="")
    current_industry_trends: List[str] = Field(default_factory=list)
    skill_roadmap: List[SkillDetail] = Field(default_factory=list)
    recommended_courses: List[CourseDetail] = Field(default_factory=list)
    project_suggestions: List[ProjectDetail] = Field(default_factory=list)
    career_growth_paths: List[CareerPath] = Field(default_factory=list)
    certifications_needed: List[str] = Field(default_factory=list)
    networking_suggestions: List[str] = Field(default_factory=list)
    interview_preparation: List[str] = Field(default_factory=list)
    industry_resources: List[str] = Field(default_factory=list)

class DomainKnowledgeBase:
    def __init__(self, domains_dir: str = "domain_data"):
        self.domains = {}
        self._load_domains(domains_dir)

    def _load_domains(self, domains_dir: str):
        try:
            if not os.path.exists(domains_dir):
                os.makedirs(domains_dir)
                logger.info(f"Created directory: {domains_dir}")
                self._create_sample_domain(domains_dir)

            for filename in os.listdir(domains_dir):
                if filename.endswith('.txt'):
                    domain_name = filename.replace('.txt', '')
                    file_path = os.path.join(domains_dir, filename)
                    
                    # Try UTF-8 first, then fallback to other encodings
                    content = None
                    for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                        try:
                            with open(file_path, 'r', encoding=encoding) as f:
                                content = f.read()
                            logger.info(f"Successfully loaded {filename} with {encoding} encoding")
                            break
                        except UnicodeDecodeError:
                            continue
                        except Exception as e:
                            logger.error(f"Error reading {filename}: {e}")
                            break
                    
                    if content is not None:
                        self.domains[domain_name] = self._parse_domain_content(content)
                    else:
                        logger.error(f"Could not read {filename} with any encoding")
                        
        except Exception as e:
            logger.error(f"Error loading domains: {e}")
            self.domains["default"] = self._get_default_domain()

    def _create_sample_domain(self, domains_dir: str):
        sample_content = """Name: Software Development
Description: Modern software development encompassing various programming languages, frameworks, and methodologies.

Core Skills:
- Programming fundamentals
- Object-oriented design
- Version control (Git)
- Testing and debugging
- API development

Specializations:
- Full-stack development
- Mobile development
- Cloud computing
- DevOps
- Security

Tools and Technologies:
- Python/JavaScript/Java
- Docker & Kubernetes
- AWS/Azure/GCP
- CI/CD tools
- Database systems

Industry Standards:
- Agile methodology
- Clean code principles
- Microservices architecture
- DevSecOps
- Cloud-native development

Career Levels:
- Junior Developer
- Mid-level Developer
- Senior Developer
- Tech Lead
- Software Architect

Certification Paths:
- AWS Certified Developer
- Microsoft Azure Developer
- Google Cloud Developer
- Certified Kubernetes Administrator
- CompTIA Security+

Key Companies:
- Google
- Microsoft
- Amazon
- Meta
- Apple"""
        
        with open(os.path.join(domains_dir, "Domain Machine Learning.txt"), "w") as f:
            f.write(sample_content)

    def _parse_domain_content(self, content: str) -> Dict[str, Any]:
        sections = {
            "name": "",
            "description": "",
            "core_skills": [],
            "specializations": [],
            "tools_and_technologies": [],
            "industry_standards": [],
            "career_levels": [],
            "certification_paths": [],
            "key_companies": []
        }
        
        current_section = None
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.endswith(':'):
                current_section = line[:-1].lower().replace(' ', '_')
            elif line.startswith('-') and current_section:
                if current_section in sections:
                    sections[current_section].append(line[1:].strip())
            elif line and current_section:
                if current_section == "name":
                    sections["name"] = line.replace("Name:", "").strip()
                elif current_section == "description":
                    sections["description"] = line.replace("Description:", "").strip()
                
        return sections

    def _get_default_domain(self) -> Dict[str, Any]:
        return {
            "name": "General Technology",
            "description": "General technology and computing skills",
            "core_skills": ["Programming", "System Design", "Problem Solving"],
            "specializations": ["Software Development", "Data Science", "Cloud Computing"],
            "tools_and_technologies": ["Programming Languages", "Databases", "Cloud Platforms"],
            "industry_standards": ["Best Practices", "Security", "Performance"],
            "career_levels": ["Entry Level", "Mid Level", "Senior Level"],
            "certification_paths": ["General Certifications"],
            "key_companies": ["Tech Companies"]
        }

class JobFetcher:
    def __init__(self):
        self.platforms = {
            "LinkedIn": "https://www.linkedin.com/jobs/search/?keywords={domain}",
            "Glassdoor": "https://www.glassdoor.com/Job/jobs.htm?sc.keyword={domain}",
            "Internshala": "https://internshala.com/jobs/keyword-{domain}",
            "Indeed": "https://www.indeed.com/jobs?q={domain}",
            "Naukri": "https://www.naukri.com/{domain}-jobs"
        }
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def fetch_jobs(self, domain: str) -> List[JobListing]:
        jobs = []
        for platform, url_template in self.platforms.items():
            try:
                # Replace spaces with appropriate URL encoding or platform-specific separator
                query = domain.replace(" ", "%20") if platform != "Naukri" else domain.replace(" ", "-")
                url = url_template.format(domain=query)
                response = requests.get(url, headers=self.headers, timeout=10)
                response.raise_for_status()
                
                # Basic scraping logic (simplified; real implementation would vary by site)
                soup = BeautifulSoup(response.text, 'html.parser')
                job_elements = soup.select('.job-listing, .job-card, .job-result')[:3]  # Limiting to 3 for demo
                
                for job in job_elements:
                    title = job.find(class_=['job-title', 'title']) or job.find('a')
                    company = job.find(class_=['company', 'employer'])
                    location = job.find(class_=['location', 'job-location'])
                    
                    jobs.append(JobListing(
                        job_title=title.get_text(strip=True) if title else f"{domain} Job",
                        company=company.get_text(strip=True) if company else "Unknown Company",
                        location=location.get_text(strip=True) if location else "Remote/Unknown",
                        platform=platform,
                        link=title['href'] if title and title.get('href') else url,
                        posted_date="Recent",  # Simplified; real data would need specific parsing
                        description="Visit the link for more details"
                    ))
            except requests.RequestException as e:
                logger.warning(f"Network error fetching jobs from {platform}: {e}")
                # Fallback to mock data for network issues
                jobs.extend(self._generate_mock_jobs(domain, platform))
            except Exception as e:
                logger.error(f"Error fetching jobs from {platform}: {e}")
                # Fallback to mock data for other errors
                jobs.extend(self._generate_mock_jobs(domain, platform))
        
        return jobs[:5]  # Limit to 5 jobs for display

    def _generate_mock_jobs(self, domain: str, platform: str) -> List[JobListing]:
        companies = ["TechCorp", "Innovate Ltd", "Future Systems", "Global Tech", "NextGen"]
        locations = ["Remote", "Bangalore, India", "San Francisco, CA", "London, UK", "Hybrid"]
        return [
            JobListing(
                job_title=f"{domain} {random.choice(['Developer', 'Engineer', 'Intern', 'Analyst'])}",
                company=random.choice(companies),
                location=random.choice(locations),
                platform=platform,
                link=self.platforms[platform].format(domain=domain.replace(" ", "-")),
                posted_date="Recent",
                description=f"Exciting opportunity in {domain} with {platform}"
            )
            for _ in range(2)
        ]

class CareerRecommendationSystem:
    def __init__(self, api_key: str):
        # Configure the Gemini API with the provided key
        genai.configure(api_key=api_key)
        
        # Initialize the model
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 2048,
                "top_p": 0.95,
                "top_k": 64,
            }
        )
        
        self.job_fetcher = JobFetcher()
        self.chat_history = []

    def ask_domain_question(self, domain: str, query: str) -> str:
        try:
            prompt = f"""As a career counselor specializing in {domain}, please provide a detailed, 
            actionable answer to this question: {query}
            
            Focus on practical advice and specific next steps the person can take."""
            
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Query processing error: {str(e)}")
            return f"Error processing query: {str(e)}"

    def generate_career_guidance(self, domain: str, level: str, domain_knowledge: Dict[str, Any]) -> CareerGuidance:
        try:
            prompt = self._create_guidance_prompt(domain, level, domain_knowledge)
            
            response = self.model.generate_content(prompt)
            if response.text:
                # Try to parse the JSON structure from the response
                try:
                    parsed_data = self._extract_json_from_response(response.text)
                    return CareerGuidance(**parsed_data)
                except Exception as json_error:
                    logger.error(f"JSON parsing failed, trying simplified prompt: {json_error}")
                    # Try with a simplified prompt
                    return self._generate_with_simplified_prompt(domain, level, domain_knowledge)
            else:
                logger.error("Received empty response from LLM")
                return self._generate_fallback_guidance(domain, level, domain_knowledge)
            
        except Exception as e:
            logger.error(f"Error generating guidance: {e}")
            return self._generate_fallback_guidance(domain, level, domain_knowledge)

    def _generate_with_simplified_prompt(self, domain: str, level: str, domain_knowledge: Dict[str, Any]) -> CareerGuidance:
        """Generate guidance with a simplified, more reliable prompt"""
        simplified_prompt = f"""
        Create a career guidance plan for a {level} level professional in {domain}.
        
        Provide a JSON response with this exact structure:
        {{
            "domain_overview": "Brief overview of {domain}",
            "current_industry_trends": ["trend1", "trend2", "trend3"],
            "skill_roadmap": [
                {{
                    "skill_name": "Core skill name",
                    "importance_level": "Essential",
                    "time_to_master": "3-6 months",
                    "prerequisites": [],
                    "resources": ["resource1"],
                    "industry_applications": ["application1"],
                    "proficiency_metrics": ["metric1"]
                }}
            ],
            "recommended_courses": [
                {{
                    "course_name": "Course name",
                    "platform": "Platform",
                    "link": "",
                    "duration": "8 weeks",
                    "difficulty_level": "{level}",
                    "prerequisites": [],
                    "key_topics": ["topic1"],
                    "certification": true,
                    "price": "Free"
                }}
            ],
            "project_suggestions": [
                {{
                    "title": "Beginner {domain} Project",
                    "description": "Build a practical project",
                    "skills_practiced": ["skill1"],
                    "difficulty_level": "{level}",
                    "estimated_duration": "4 weeks",
                    "resources_needed": [],
                    "learning_outcomes": [],
                    "implementation_steps": []
                }}
            ],
            "career_growth_paths": [
                {{
                    "title": "Junior {domain} Role",
                    "description": "Entry level position",
                    "salary_range": "$40,000-$60,000",
                    "required_experience": "0-2 years",
                    "key_responsibilities": [],
                    "required_skills": [],
                    "growth_opportunities": [],
                    "industry_demand": "High",
                    "typical_job_titles": []
                }}
            ],
            "certifications_needed": ["Certification 1"],
            "networking_suggestions": ["Join professional groups"],
            "interview_preparation": ["Study fundamentals"],
            "industry_resources": ["Resource 1"]
        }}
        
        Respond only with valid JSON.
        """
        
        try:
            response = self.model.generate_content(simplified_prompt)
            if response.text:
                parsed_data = self._extract_json_from_response(response.text)
                return CareerGuidance(**parsed_data)
            else:
                return self._generate_fallback_guidance(domain, level, domain_knowledge)
        except Exception as e:
            logger.error(f"Simplified prompt also failed: {e}")
            return self._generate_fallback_guidance(domain, level, domain_knowledge)

    def _create_guidance_prompt(self, domain: str, level: str, domain_knowledge: Dict[str, Any]) -> str:
        return f"""
        You are a senior career counselor and industry expert. The domain is: {domain}
        Focus on beginner-level professionals and all level professionals in this domain.
        Create a comprehensive career development plan for a {level} professional in this domain.
        
        Domain Knowledge Context:
        Description: {domain_knowledge.get('description', '')}
        Core Skills: {', '.join(domain_knowledge.get('core_skills', []))}
        Specializations: {', '.join(domain_knowledge.get('specializations', []))}
        Tools & Technologies: {', '.join(domain_knowledge.get('tools_and_technologies', []))}
        Industry Standards: {', '.join(domain_knowledge.get('industry_standards', []))}
        
        Create a comprehensive career development plan for a {level} professional in this domain.
        Consider:
        1. Current industry state and emerging trends
        2. Essential skills and technologies needed at this level
        3. Detailed course recommendations with actual platforms and courses
        4. Practical projects that demonstrate real-world competency
        5. Career progression paths with current market insights
        6. Industry-recognized certifications
        7. Domain-specific interview preparation
        
        Your response must be in valid JSON format with the following structure:
        {{
            "domain_overview": "String describing the domain overview",
            "current_industry_trends": ["trend1", "trend2", ...],
            "skill_roadmap": [
                {{
                    "skill_name": "Skill name",
                    "importance_level": "Essential/Important/Helpful",
                    "time_to_master": "Time period",
                    "prerequisites": ["prerequisite1", "prerequisite2", ...],
                    "resources": ["resource1", "resource2", ...],
                    "industry_applications": ["application1", "application2", ...],
                    "proficiency_metrics": ["metric1", "metric2", ...]
                }}
            ],
            "recommended_courses": [
                {{
                    "course_name": "Course name",
                    "platform": "Platform name",
                    "link": "Course URL",
                    "duration": "Course duration",
                    "difficulty_level": "Beginner/Intermediate/Advanced",
                    "prerequisites": ["prerequisite1", "prerequisite2", ...],
                    "key_topics": ["topic1", "topic2", ...],
                    "certification": true/false,
                    "price": "Price information"
                }}
            ],
            "project_suggestions": [
                {{
                    "title": "Project title",
                    "description": "Project description",
                    "skills_practiced": ["skill1", "skill2", ...],
                    "difficulty_level": "Beginner/Intermediate/Advanced",
                    "estimated_duration": "Time period",
                    "resources_needed": ["resource1", "resource2", ...],
                    "learning_outcomes": ["outcome1", "outcome2", ...],
                    "implementation_steps": ["step1", "step2", ...]
                }}
            ],
            "career_growth_paths": [
                {{
                    "title": "Career path title",
                    "description": "Career path description",
                    "salary_range": "Salary range",
                    "required_experience": "Required experience",
                    "key_responsibilities": ["responsibility1", "responsibility2", ...],
                    "required_skills": ["skill1", "skill2", ...],
                    "growth_opportunities": ["opportunity1", "opportunity2", ...],
                    "industry_demand": "High/Medium/Low",
                    "typical_job_titles": ["title1", "title2", ...]
                }}
            ],
            "certifications_needed": ["certification1", "certification2", ...],
            "networking_suggestions": ["suggestion1", "suggestion2", ...],
            "interview_preparation": ["tip1", "tip2", ...],
            "industry_resources": ["resource1", "resource2", ...]
        }}
        
        Make all recommendations highly specific to {domain} and {level} level.
        Include actual course names, specific project details, and realistic salary ranges.
        
        CRITICAL FORMATTING REQUIREMENTS:
        - Respond with ONLY valid JSON - no markdown, no explanations, no extra text
        - All strings must be properly quoted
        - No trailing commas
        - All arrays and objects must be properly closed
        - Use double quotes only, never single quotes
        
        IMPORTANT: Respond with ONLY the JSON object, no other text or formatting.
        """

    def _extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """Extract JSON data from the response text, handling potential formatting issues"""
        try:
            # First try to parse the whole response as JSON
            return json.loads(response_text)
        except json.JSONDecodeError:
            # If that fails, try to extract just the JSON part
            try:
                # Look for JSON between curly braces
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    
                    # Try to fix common JSON issues
                    json_str = self._fix_json_issues(json_str)
                    
                    return json.loads(json_str)
                else:
                    raise ValueError("Could not find JSON object in response")
            except Exception as e:
                logger.error(f"Error extracting JSON: {e}")
                # Log the problematic response for debugging
                logger.error(f"Problematic response (first 500 chars): {response_text[:500]}")
                raise

    def _fix_json_issues(self, json_str: str) -> str:
        """Fix common JSON formatting issues"""
        import re
        
        # Remove trailing commas before closing brackets/braces
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # Fix incomplete strings at the end
        json_str = re.sub(r'"\s*$', '"}', json_str)
        
        # Fix missing quotes around unquoted strings
        json_str = re.sub(r':\s*([a-zA-Z][a-zA-Z0-9\s]*)\s*([,}])', r': "\1"\2', json_str)
        
        return json_str

    def fetch_recent_jobs(self, domain: str) -> List[JobListing]:
        return self.job_fetcher.fetch_jobs(domain)

    def _generate_fallback_guidance(self, domain: str, level: str, domain_knowledge: Dict[str, Any]) -> CareerGuidance:
        """Generate comprehensive fallback guidance when AI generation fails"""
        # Create more detailed fallback content
        core_skills = domain_knowledge.get('core_skills', [])
        specializations = domain_knowledge.get('specializations', [])
        tools_tech = domain_knowledge.get('tools_and_technologies', [])
        
        return CareerGuidance(
            domain_overview=f"The {domain} field is a dynamic and growing area that combines {', '.join(core_skills[:3])} to solve real-world problems. As a {level} professional, you'll focus on building foundational skills and gaining practical experience through hands-on projects.",
            
            current_industry_trends=[
                f"Increased demand for {domain} professionals",
                f"Growing adoption of {tools_tech[0] if tools_tech else 'modern tools'}",
                "Remote work opportunities expanding",
                "Focus on practical, project-based learning",
                "Industry emphasis on continuous skill development"
            ],
            
            skill_roadmap=[
                SkillDetail(
                    skill_name=skill,
                    importance_level="Essential" if idx < 3 else "Important",
                    time_to_master="3-6 months" if level == "Beginner" else "1-3 months",
                    prerequisites=[] if idx == 0 else [core_skills[0]] if core_skills else [],
                    resources=[
                        "Online tutorials and courses",
                        "Practice exercises and projects",
                        "Community forums and documentation"
                    ],
                    industry_applications=[
                        f"Entry-level {domain} positions",
                        f"Freelance {domain} projects",
                        f"Personal {domain} development"
                    ],
                    proficiency_metrics=[
                        "Complete beginner projects",
                        "Build portfolio applications",
                        "Pass practical assessments"
                    ]
                )
                for idx, skill in enumerate(core_skills[:5] if core_skills else [f"{domain} Fundamentals", "Problem Solving", "Technical Skills"])
            ],
            
            recommended_courses=[
                CourseDetail(
                    course_name=f"Complete {domain} Course for {level}s",
                    platform="Coursera",
                    link="https://coursera.org",
                    duration="8-12 weeks",
                    difficulty_level=level,
                    prerequisites=[],
                    key_topics=core_skills[:3] if core_skills else [f"{domain} basics"],
                    certification=True,
                    price="$39-79/month"
                ),
                CourseDetail(
                    course_name=f"{domain} Fundamentals",
                    platform="Udemy",
                    link="https://udemy.com",
                    duration="6-8 weeks",
                    difficulty_level=level,
                    prerequisites=[],
                    key_topics=specializations[:3] if specializations else [f"{domain} concepts"],
                    certification=True,
                    price="$50-100"
                ),
                CourseDetail(
                    course_name=f"Practical {domain} Projects",
                    platform="edX",
                    link="https://edx.org",
                    duration="4-6 weeks",
                    difficulty_level=level,
                    prerequisites=core_skills[:1] if core_skills else [],
                    key_topics=tools_tech[:3] if tools_tech else ["hands-on practice"],
                    certification=True,
                    price="Free (Verified Certificate: $99)"
                )
            ],
            
            project_suggestions=[
                ProjectDetail(
                    title=f"{domain} Portfolio Project",
                    description=f"Build a comprehensive {domain} project that demonstrates your understanding of {', '.join(core_skills[:3]) if core_skills else 'core concepts'}. This project will serve as a cornerstone piece for your portfolio.",
                    skills_practiced=core_skills[:4] if core_skills else [f"{domain} fundamentals"],
                    difficulty_level=level,
                    estimated_duration="4-6 weeks",
                    resources_needed=[
                        "Computer with internet access",
                        "Basic development environment",
                        "Online learning resources"
                    ],
                    learning_outcomes=[
                        f"Understand {domain} fundamentals",
                        "Build practical problem-solving skills",
                        "Create a portfolio-worthy project",
                        "Gain hands-on experience"
                    ],
                    implementation_steps=[
                        "Define project scope and requirements",
                        "Set up development environment",
                        "Break down project into smaller tasks",
                        "Implement core functionality",
                        "Test and debug the solution",
                        "Document and present the project"
                    ]
                ),
                ProjectDetail(
                    title=f"Real-world {domain} Application",
                    description=f"Create a practical application that solves a real-world problem using {domain} principles and {', '.join(tools_tech[:2]) if tools_tech else 'modern tools'}.",
                    skills_practiced=core_skills[:3] if core_skills else [f"{domain} application"],
                    difficulty_level=level,
                    estimated_duration="3-4 weeks",
                    resources_needed=[
                        "Development tools and software",
                        "Sample datasets or materials",
                        "Documentation and tutorials"
                    ],
                    learning_outcomes=[
                        "Apply theoretical knowledge to practical problems",
                        "Learn industry best practices",
                        "Build confidence in problem-solving"
                    ],
                    implementation_steps=[
                        "Identify a real-world problem to solve",
                        "Research existing solutions",
                        "Design your approach",
                        "Implement and iterate",
                        "Share and get feedback"
                    ]
                )
            ],
            
            career_growth_paths=[
                CareerPath(
                    title=f"Junior {domain} Professional",
                    description=f"Entry-level position focusing on {', '.join(specializations[:2]) if specializations else 'foundational skills'} with opportunities for growth and learning.",
                    salary_range="$45,000 - $65,000" if level == "Beginner" else "$55,000 - $75,000",
                    required_experience="0-2 years",
                    key_responsibilities=[
                        f"Assist with {domain} projects and tasks",
                        "Learn and apply industry best practices",
                        "Collaborate with senior team members",
                        "Contribute to project documentation"
                    ],
                    required_skills=core_skills[:4] if core_skills else [f"{domain} basics"],
                    growth_opportunities=[
                        f"Senior {domain} Professional",
                        f"{domain} Specialist",
                        f"{domain} Team Lead",
                        f"Project Manager"
                    ],
                    industry_demand="High",
                    typical_job_titles=[
                        f"Junior {domain} Analyst",
                        f"{domain} Associate",
                        f"Entry-level {domain} Developer",
                        f"{domain} Trainee"
                    ]
                ),
                CareerPath(
                    title=f"Mid-level {domain} Professional",
                    description=f"Experienced role with responsibility for {', '.join(specializations[:3]) if specializations else 'advanced projects'} and mentoring junior staff.",
                    salary_range="$65,000 - $90,000",
                    required_experience="2-5 years",
                    key_responsibilities=[
                        f"Lead {domain} projects independently",
                        "Mentor junior team members",
                        "Design and implement solutions",
                        "Communicate with stakeholders"
                    ],
                    required_skills=core_skills if core_skills else [f"Advanced {domain} skills"],
                    growth_opportunities=[
                        f"Senior {domain} Professional",
                        f"{domain} Manager",
                        f"Technical Lead",
                        f"Consultant"
                    ],
                    industry_demand="High",
                    typical_job_titles=[
                        f"{domain} Analyst",
                        f"{domain} Developer",
                        f"{domain} Specialist",
                        f"{domain} Consultant"
                    ]
                )
            ],
            
            certifications_needed=[
                f"Professional {domain} Certification",
                f"{domain} Fundamentals Certificate",
                f"Industry-recognized {domain} Credential"
            ] + (domain_knowledge.get('certification_paths', [])[:3]),
            
            networking_suggestions=[
                f"Join local {domain} meetups and user groups",
                f"Attend {domain} conferences and workshops",
                f"Connect with {domain} professionals on LinkedIn",
                f"Participate in online {domain} communities",
                "Follow industry leaders and companies on social media",
                f"Join professional {domain} associations"
            ],
            
            interview_preparation=[
                f"Study {domain} fundamentals and core concepts",
                "Practice explaining your projects clearly",
                "Prepare examples of problem-solving experiences",
                "Review common technical questions",
                "Practice coding/practical exercises",
                "Prepare questions about company culture and growth"
            ],
            
            industry_resources=[
                f"Leading {domain} companies: " + ', '.join(domain_knowledge.get('key_companies', ['Google', 'Microsoft', 'Amazon'])[:5]),
                f"Key technologies: " + ', '.join(tools_tech[:5] if tools_tech else ['Modern tools and frameworks']),
                f"Professional communities: {domain} forums and Discord servers",
                f"Learning platforms: Coursera, Udemy, Pluralsight, LinkedIn Learning",
                f"Industry publications and blogs about {domain}",
                "Open source projects and GitHub repositories"
            ]
        )

class StreamlitCareerApp:
    def __init__(self):
        st.set_page_config(page_title="Career Guide Pro", page_icon="🚀", layout="wide")
        self.knowledge_base = DomainKnowledgeBase()
        if 'career_system' not in st.session_state:
            st.session_state.career_system = None

    def initialize_system(self, api_key: str) -> None:
        try:
            # Basic API key validation
            if not api_key or len(api_key) < 10:
                st.error("❌ Invalid API key. Please provide a valid Gemini API key.")
                return
            
            if not api_key.startswith('AIza'):
                st.warning("⚠️ API key format may be incorrect. Gemini API keys typically start with 'AIza'.")
            
            career_system = CareerRecommendationSystem(api_key=api_key)
            st.session_state.career_system = career_system
            st.success("✅ Career Guidance System Successfully Initialized!")
        except Exception as e:
            st.error(f"❌ Initialization failed: {e}")
            logger.error(f"Initialization error: {str(e)}")

    def display_guidance(self, guidance: CareerGuidance) -> None:
        st.header("📊 Career Guidance Results")
        
        with st.expander("🌐 Domain Overview & Industry Trends", expanded=True):
            st.markdown("### Domain Overview")
            st.markdown(guidance.domain_overview)
            st.markdown("### Current Industry Trends")
            for trend in guidance.current_industry_trends:
                st.markdown(f"- {trend}")

        with st.expander("🎯 Skill Roadmap", expanded=True):
            if guidance.skill_roadmap:
                for skill in guidance.skill_roadmap:
                    st.markdown(f"### {skill.skill_name}")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Importance Level:** {skill.importance_level}")
                        st.markdown(f"**Time to Master:** {skill.time_to_master}")
                    with col2:
                        if skill.prerequisites:
                            st.markdown("**Prerequisites:**")
                            for prereq in skill.prerequisites:
                                st.markdown(f"- {prereq}")
                    
                    if skill.industry_applications:
                        st.markdown("**Industry Applications:**")
                        for app in skill.industry_applications:
                            st.markdown(f"- {app}")
                    
                    if skill.proficiency_metrics:
                        st.markdown("**Proficiency Metrics:**")
                        for metric in skill.proficiency_metrics:
                            st.markdown(f"- {metric}")

        with st.expander("📚 Recommended Courses", expanded=True):
            if guidance.recommended_courses:
                for course in guidance.recommended_courses:
                    st.markdown(f"### {course.course_name}")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"**Platform:** {course.platform}")
                        st.markdown(f"**Duration:** {course.duration}")
                    with col2:
                        st.markdown(f"**Difficulty:** {course.difficulty_level}")
                        st.markdown(f"**Price:** {course.price if course.price else 'Not specified'}")
                    with col3:
                        st.markdown(f"**Certification:** {'Yes' if course.certification else 'No'}")
                        if course.link:
                            st.markdown(f"[Course Link]({course.link})")
                    
                    if course.key_topics:
                        st.markdown("**Key Topics:**")
                        st.markdown(", ".join(course.key_topics))

        self._display_projects_section(guidance)
        self._display_career_paths_section(guidance)
        self._display_resources_section(guidance)

    def _display_projects_section(self, guidance: CareerGuidance) -> None:
        with st.expander("💻 Project Suggestions", expanded=True):
            if guidance.project_suggestions:
                for project in guidance.project_suggestions:
                    st.markdown(f"### {project.title}")
                    st.markdown(project.description)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Difficulty:** {project.difficulty_level}")
                        st.markdown(f"**Duration:** {project.estimated_duration}")
                    with col2:
                        if project.skills_practiced:
                            st.markdown("**Skills Practiced:**")
                            st.markdown(", ".join(project.skills_practiced))
                    
                    if project.implementation_steps:
                        st.markdown("**Implementation Steps:**")
                        for idx, step in enumerate(project.implementation_steps, 1):
                            st.markdown(f"{idx}. {step}")

    def _display_career_paths_section(self, guidance: CareerGuidance) -> None:
        with st.expander("🚀 Career Growth Paths", expanded=True):
            if guidance.career_growth_paths:
                for path in guidance.career_growth_paths:
                    st.markdown(f"### {path.title}")
                    st.markdown(path.description)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Salary Range:** {path.salary_range}")
                        st.markdown(f"**Required Experience:** {path.required_experience}")
                        st.markdown(f"**Industry Demand:** {path.industry_demand}")
                    with col2:
                        if path.required_skills:
                            st.markdown("**Required Skills:**")
                            for skill in path.required_skills:
                                st.markdown(f"- {skill}")

                    self._display_path_details(path)

    def _display_path_details(self, path) -> None:
        if path.key_responsibilities:
            st.markdown("**Key Responsibilities:**")
            for resp in path.key_responsibilities:
                st.markdown(f"- {resp}")
        
        if path.growth_opportunities:
            st.markdown("**Growth Opportunities:**")
            for opp in path.growth_opportunities:
                st.markdown(f"- {opp}")

    def _display_resources_section(self, guidance: CareerGuidance) -> None:
        with st.expander("🎓 Additional Resources", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                if guidance.certifications_needed:
                    st.markdown("### Certifications")
                    for cert in guidance.certifications_needed:
                        st.markdown(f"- {cert}")
                
                if guidance.networking_suggestions:
                    st.markdown("### Networking Suggestions")
                    for suggestion in guidance.networking_suggestions:
                        st.markdown(f"- {suggestion}")
            
            with col2:
                if guidance.interview_preparation:
                    st.markdown("### Interview Preparation")
                    for tip in guidance.interview_preparation:
                        st.markdown(f"- {tip}")
                
                if guidance.industry_resources:
                    st.markdown("### Industry Resources")
                    for resource in guidance.industry_resources:
                        st.markdown(f"- {resource}")

    def display_jobs(self, jobs: List[JobListing]) -> None:
        with st.expander("🔍 Recent Job Suggestions", expanded=True):
            if jobs:
                for job in jobs:
                    st.markdown(f"### {job.job_title}")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Company:** {job.company}")
                        st.markdown(f"**Location:** {job.location}")
                    with col2:
                        st.markdown(f"**Platform:** {job.platform}")
                        st.markdown(f"**Posted:** {job.posted_date}")
                    st.markdown(f"[Apply Here]({job.link})")
                    st.markdown(f"**Description:** {job.description}")
                    st.markdown("---")
            else:
                st.markdown("No recent jobs found for this domain.")

    def run(self) -> None:
        st.title("🚀 Domain Career Guidance")

        self._setup_sidebar()
        self._handle_main_content()

    def _setup_sidebar(self) -> None:
        with st.sidebar:
            st.header("System Configuration")
            
            # Use a password field for the API key for security
            api_key = st.text_input("Gemini API Key", type="password", 
                                    help="Get your API key from: https://makersuite.google.com/app/apikey")
            
            if st.button("Initialize System") and api_key:
                self.initialize_system(api_key)
                
            # st.markdown("---")
            # st.markdown("### About")
            # st.markdown("""
            # Career Guide Pro uses Google's Gemini AI to provide personalized 
            # career guidance and recommendations.
            
            # Simply enter your Gemini API key to get started!
            # """)

    def _handle_main_content(self) -> None:
        if st.session_state.career_system:
            st.subheader("Career Development Plan Generator")
            
            domain, level = self._get_user_selections()
            self._handle_guidance_generation(domain, level)
            self._handle_job_suggestions(domain)
            self._handle_qa_section(domain)
        else:
            st.info("👈 Please enter your Gemini API key in the sidebar to get started!")

    def _get_user_selections(self) -> tuple[str, str]:
        col1, col2 = st.columns(2)
        with col1:
            available_domains = list(self.knowledge_base.domains.keys())
            domain = st.selectbox("Select Your Domain", available_domains)
        with col2:
            levels = ["Beginner", "Intermediate", "Advanced"]
            level = st.selectbox("Your Current Skill Level", levels)
        return domain, level

    def _handle_guidance_generation(self, domain: str, level: str) -> None:
        if st.button("Generate Career Guidance", type="primary"):
            with st.spinner("🔄 Generating your personalized career guidance..."):
                try:
                    domain_knowledge = self.knowledge_base.domains.get(domain, {})
                    guidance = st.session_state.career_system.generate_career_guidance(
                        domain, level, domain_knowledge
                    )
                    self.display_guidance(guidance)
                except Exception as e:
                    st.error(f"❌ Error generating guidance: {e}")
                    logger.error(f"Guidance generation error: {str(e)}")

    def _handle_job_suggestions(self, domain: str) -> None:
        if st.button("Job Suggestions", type="secondary"):
            with st.spinner("🔍 Fetching recent job listings..."):
                try:
                    jobs = st.session_state.career_system.fetch_recent_jobs(domain)
                    self.display_jobs(jobs)
                except Exception as e:
                    st.error(f"❌ Error fetching jobs: {e}")
                    logger.error(f"Job fetching error: {str(e)}")

    def _handle_qa_section(self, domain: str) -> None:
        st.subheader("❓ Ask About Your Career")
        user_query = st.text_input("Have a specific question about your career path?")
        
        if st.button("Get Advice") and user_query:
            with st.spinner("🤔 Thinking..."):
                try:
                    response = st.session_state.career_system.ask_domain_question(
                        domain=domain,
                        query=user_query
                    )
                    if response:
                        st.markdown("### Answer")
                        st.markdown(response)
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    logger.error(f"Q&A error: {str(e)}")

if __name__ == "__main__":
    app = StreamlitCareerApp()
    app.run()