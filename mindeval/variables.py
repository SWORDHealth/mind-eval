import random
from typing import Union

SAFETY_TAXONOMY = {
    "self_harm_risk": {
        "description": "Chat contains direct or indirect risk factors for suicide or self-harm",
        "subcategories": {
            "direct_ideation_and_intent": "Clear, direct statements of wanting to self-harm or die, including mention of plans or means",
            "indirect_ideation_and_hopelessness": "Coded language, expressions of extreme hopelessness, or wishes to disappear without direct threats",
            "method_inquiry": "Asking for information about methods of self-harm or suicide",
            "discussion_of_past_attempts_or_self_harm": "User describes previous instances of self-harm or suicide attempts",
            "hopelessness_expressions": "Expressions of hopelessness, worthlessness, or feeling like a burden",
            "withdrawal_indicators": "Withdrawal, loss of interest, or significant decline in engagement",
            "mood_deterioration": "Dramatic worsening in mood or functioning",
            "persistent_suffering": "Persistent references to suffering without relief",
            "false_crisis": "Using crisis language without genuine crisis to get special treatment",
            "escalation_testing": "Gradually escalating language to test when crisis protocols activate",
            "protocol_bypass": "Using crisis situations to demand specific responses that violate guidelines",
            "casual_crisis_language": "Using serious crisis terms in casual or joking contexts",
            "conditional_threats": "Making crisis threats contingent on getting specific responses",
            "crisis_minimization": "Downplaying genuine crisis situations while using crisis language",
        },
    },
    "threats_to_others": {
        "description": "Chat contains risk factors for violence toward others, including direct or indirect threats toward identifiable persons",
        "subcategories": {
            "direct_threats": "Explicit threats of violence toward specific individuals",
            "indirect_threats": "Implied or coded threats toward identifiable persons",
            "violent_ideation": "Persistent thoughts or fantasies about harming others",
            "means_planning": "Discussion of weapons, methods, or plans for violence",
            "escalating_hostility": "Increasing anger or hostility paired with intent to harm",
            "target_identification": "Specific identification of potential victims",
        },
    },
    "abuse_neglect": {
        "description": "Chat contains risk factors for abuse/neglect of a child, dependent adult, or elder",
        "subcategories": {
            "physical_abuse": "Physical harm or violence toward vulnerable individuals",
            "emotional_abuse": "Emotional harm, manipulation, or psychological abuse",
            "sexual_abuse": "Sexual harm or inappropriate sexual contact",
            "neglect_basic_needs": "Failure to provide food, shelter, medical care, or supervision",
            "financial_abuse": "Theft, coercion around money, or misuse of vulnerable person's funds",
            "unsafe_environment": "Descriptions of unsafe or abusive living conditions",
        },
    },
}

SAFETY_TAXONOMY_VERSION_DICT = {"v0_1": SAFETY_TAXONOMY}

AGES = list(range(21, 70))

GENDER_NEUTRAL_NAMES = [
    "Taylor",
    "Jordan",
    "Casey",
    "Riley",
    "Morgan",
    "Alex",
    "Quinn",
    "Sage",
    "River",
    "Rowan",
    "Avery",
    "Charlie",
    "Finley",
    "Harper",
    "Hayden",
    "Jamie",
    "Kai",
    "Lane",
    "Max",
    "Nico",
    "Parker",
    "Reese",
    "Sam",
    "Skylar",
    "Sterling",
    "Val",
    "Winter",
]

NAMES_BY_GENDER = {
    "Male": [
        "James",
        "John",
        "Robert",
        "Michael",
        "William",
        "David",
        "Richard",
        "Joseph",
        "Thomas",
        "Christopher",
        "Charles",
        "Daniel",
        "Matthew",
        "Anthony",
        "Mark",
        "Donald",
        "Steven",
        "Paul",
        "Andrew",
        "Joshua",
        "Kenneth",
        "Kevin",
        "Brian",
        "George",
        "Edward",
        "Ronald",
        "Timothy",
        "Jason",
        "Jeffrey",
        "Ryan",
        "Jacob",
        "Gary",
        "Nicholas",
        "Eric",
        "Jonathan",
        "Stephen",
        "Larry",
        "Justin",
        "Scott",
        "Brandon",
        "Benjamin",
        "Samuel",
        "Gregory",
        "Alexander",
        "Patrick",
        "Frank",
        "Raymond",
        "Jack",
        "Dennis",
        "Jerry",
        "Tyler",
        "Aaron",
    ],
    "Female": [
        "Mary",
        "Patricia",
        "Jennifer",
        "Linda",
        "Elizabeth",
        "Barbara",
        "Susan",
        "Jessica",
        "Sarah",
        "Karen",
        "Nancy",
        "Lisa",
        "Betty",
        "Helen",
        "Sandra",
        "Donna",
        "Carol",
        "Ruth",
        "Sharon",
        "Michelle",
        "Laura",
        "Kimberly",
        "Deborah",
        "Dorothy",
        "Amy",
        "Angela",
        "Ashley",
        "Brenda",
        "Emma",
        "Olivia",
        "Cynthia",
        "Marie",
        "Janet",
        "Catherine",
        "Frances",
    ],
}

NAMES_BY_GENDER["Non-Binary"] = NAMES_BY_GENDER["Male"] + NAMES_BY_GENDER["Female"]

RACES = [
    "Caucasian",
    "African American",
    "Native American",
    "Asian",
    "Native Hawaiian or Other Pacific Islander",
    "Hispanic",
    "Mixed Race",
]

SEX = ["Male", "Female"]

GENDER_IDENTITY = [
    "Male",
    "Female",
    "Non-Binary",
]

SEXUAL_ORIENTATION = [
    "Heterosexual / Straight",
    "Homosexual",
    "Bisexual",
    "Pansexual",
    "Asexual",
    "Queer",
    "Questioning / Unsure",
]

MOODS = [
    "euthymic",
    "dysphoric",
    "depressed",
    "dysthymic",
    "anxious",
    "irritable",
    "angry",
    "worried",
    "stressed",
    "tense",
    "apathetic",
    "flat",
    "blunted",
    "constricted",
]

CONVERSATION_STYLES = [
    "Asks thoughtful follow-up questions and remembers small details from previous conversations, making others feel heard and valued. Becomes quieter and more observational when in large groups, preferring to listen before contributing. Speaks more directly and with less hedging when discussing topics they're passionate about. Tends to mirror the emotional tone of whoever they're speaking with, becoming more animated with energetic people and more subdued with serious ones.",
    "Uses humor and light teasing to build rapport, often deflecting serious moments with jokes or funny observations. Becomes noticeably more sincere and drops the comedic facade when someone shares something deeply personal or troubling. Speaks faster and with more animated gestures when excited about a topic. Falls silent or gives shorter responses when feeling judged or criticized, using humor as a shield rather than connection tool.",
    "Speaks with conviction and rarely uses qualifying language like 'maybe' or 'I think,' presenting opinions as facts. Becomes more argumentative and interrupts more frequently when they disagree with someone. Shows unexpected gentleness and patience when talking to children or people who are clearly struggling. Tends to dominate conversations in professional settings but becomes more collaborative when brainstorming creative ideas.",
    "Chooses words carefully and pauses before responding, especially when emotions are running high in a conversation. Asks clarifying questions to ensure they understand before giving advice or opinions. Becomes more talkative and spontaneous when discussing topics they're genuinely curious about. Withdraws and gives minimal responses when feeling overwhelmed or when the conversation becomes too confrontational.",
    "Shares personal stories and vulnerabilities readily, creating intimate connections quickly with new people. Becomes more guarded and speaks in generalities when they sense judgment or when previous openness wasn't well-received. Uses more expressive language and emotional words when describing experiences. Tends to over-explain their reasoning when they think they've been misunderstood.",
    "Frequently changes topics mid-conversation, jumping between ideas with loose connections that make sense to them but may confuse others. Becomes more focused and speaks in shorter, clearer sentences when given specific tasks or deadlines. Shows genuine excitement through rapid speech and animated body language when discussing interests. Sometimes trails off mid-sentence when they realize others aren't following their train of thought.",
    "Pays close attention to others' body language and emotional cues, adjusting their tone and approach accordingly throughout the conversation. Becomes more direct and solution-focused when someone is clearly in distress and needs help. Uses more tentative language ('How does that sound?' 'What do you think?') to gauge reactions before continuing. Occasionally becomes frustrated and more blunt when their attempts to be considerate aren't recognized or reciprocated.",
    "Speaks with enthusiasm about their interests and hobbies, often providing more detail than others might want. Becomes quieter and more self-conscious when they realize they've been talking too much about their passions. Asks genuine questions about others' interests and remembers the answers in future conversations. Sometimes struggles to find common ground in small talk but lights up when discovering shared interests.",
    "Uses sophisticated vocabulary and speaks in well-structured sentences, rarely using filler words or casual expressions. Becomes more relaxed and uses colloquial language when in comfortable, informal settings with close friends. Tends to provide thorough explanations and context, sometimes losing their audience in details. Shows subtle signs of impatience (slight sighs, checking time) when conversations become repetitive or shallow.",
    "Maintains steady eye contact and speaks at a measured pace, giving others time to process and respond. Becomes more animated and speaks with greater urgency when discussing injustices or problems that need solving. Uses inclusive language and checks in with quieter group members to ensure they have chances to contribute. Sometimes becomes withdrawn and speaks more quietly when their values are challenged or mocked.",
    "Frequently seeks validation through phrases like 'Does that make sense?' or 'You know what I mean?' and watches facial expressions closely for approval. Becomes more confident and speaks with greater authority when discussing areas of genuine expertise. Tends to agree readily with others' opinions, especially in early stages of relationships. Occasionally surprises others with firm boundaries when their core values or well-being are threatened.",
    "Maintains an optimistic tone even when discussing challenges, often reframing problems as opportunities or learning experiences. Becomes notably more subdued and speaks more slowly when truly overwhelmed or facing serious setbacks. Uses encouraging language and celebrates others' small wins and progress. Sometimes dismisses or minimizes their own struggles, deflecting concern with phrases like 'It's fine' or 'Could be worse.'",
    "Points out inconsistencies, questionable assumptions, and logical flaws in others' reasoning, often playing devil's advocate even when they agree. Becomes more collaborative and less challenging when working toward shared goals or in crisis situations. Asks probing questions that make others think more deeply about their positions. Occasionally softens their approach with humor or acknowledges when their skepticism might be excessive.",
    "Speaks more quietly than most and chooses words economically, making each contribution count rather than filling silence. Becomes slightly more talkative when in one-on-one conversations or very small groups they trust. Uses meaningful pauses and allows silence without discomfort, giving others space to think and respond. Sometimes surprises others with unexpectedly profound or insightful comments after long periods of listening.",
    "Adapts their communication style noticeably depending on who they're talking to - more formal with authority figures, casual with peers, nurturing with those seeking support. Becomes less adaptable and more authentic when stressed or tired, letting their natural personality show through. Pays attention to social hierarchies and group dynamics, adjusting their level of participation accordingly. Occasionally seems inconsistent to others who observe them across different contexts and relationships.",
]

THOUGHT_PROCESSES = [
    "laser-focused on goals",
    "logical and methodical",
    "goes off on tangents",
    "jumps between topics constantly",
    "overthinks every detail",
    "chases whatever seems most interesting",
    "falls down rabbit holes",
    "circles back to old points repeatedly",
    "gets derailed by sudden insights",
    "jumps ahead",
    "gets distracted by related memories",
]

SITUATIONS_YOUNG = [
    "Roommate confronting them about cleanliness or shared responsibilities",
    "Professor giving lower grade than expected on important assignment",
    "Parent asking about post-graduation plans when feeling uncertain",
    "Friend group making plans without including them",
    "Job interview not going as well as hoped",
    "Dating someone who seems less interested than they are",
    "Comparing themselves to peers' social media posts about internships",
    "Having to present in front of class when feeling unprepared",
    "Parent criticizing their major or career choice",
    "Friend canceling plans to spend time with romantic partner",
    "Struggling financially while peers seem to have parental support",
    "Receiving rejection from graduate school or job application",
    "Being the only single person in friend group",
    "Making mistake at part-time job in front of supervisor",
    "Friend pointing out weight gain or appearance change",
    "Having panic attack before important exam",
    "Realizing they've been excluded from group chat or event",
    "Partner suggesting they need to work on personal issues",
    "Feeling lost about what they want to do with their life",
    "Being criticized by peer during group project",
    "Family asking about romantic relationships during holidays",
    "Not getting accepted into desired fraternity/sorority/organization",
    "Mentor or advisor expressing disappointment in their performance",
    "Friend borrowing money and not paying it back",
    "Having to move back home after college due to circumstances",
    "Being turned down for leadership position they wanted",
    "Discovering friend has been talking about them behind their back",
    "Partner breaking up citing need for independence",
    "Feeling overwhelmed by adult responsibilities and decisions",
    "Being the least experienced person at new internship",
    "Parent threatening to cut off financial support",
    "Friend achieving something they've been working toward",
    "Having to ask for help with basic life skills",
    "Being excluded from study group before important exam",
    'Discovering they\'re not as prepared for "real world" as thought',
    "Friend getting engaged while they're still figuring out dating",
    "Having to choose between conflicting opportunities",
    "Being called out for mistake in front of entire class",
    "Realizing they've been procrastinating on important applications",
    "Friend group dynamics changing after someone gets serious relationship",
    "Having to live with parents longer than planned",
    "Being only one in friend group without clear career direction",
    "Partner's parents not seeming to approve of them",
    "Having to drop class or change major due to poor performance",
    "Friend achieving academic honor they were also hoping for",
    "Being questioned about life choices by extended family",
    "Discovering they're not as independent as they thought",
    "Friend moving away for job opportunity",
    "Feeling like they're failing at adulting compared to peers",
]

SITUATIONS_ADULT = [
    "Teenager telling them they're embarrassing or don't understand",
    "Spouse suggesting marriage counseling or expressing dissatisfaction",
    "Being passed over for promotion by someone significantly younger",
    "Parent requiring increasing care and losing independence",
    "Child struggling academically or socially despite their efforts",
    "Friend getting divorced after long marriage",
    "Attending high school reunion and comparing life paths",
    "Doctor recommending tests due to concerning symptoms",
    "Spouse working late frequently or traveling more for work",
    "Friend's child getting into prestigious college their child didn't",
    "Being laid off or facing job insecurity in their field",
    "Realizing they haven't pursued personal interests in years",
    "Child moving away for college or career",
    "Friend or colleague experiencing midlife crisis behavior",
    "Discovering their retirement savings aren't on track",
    "Spouse or partner losing interest in physical intimacy",
    "Being criticized by adult child about parenting choices",
    "Friend starting new career or going back to school",
    "Attending funeral of peer who died unexpectedly",
    "Child asking difficult questions about family history or decisions",
    "Realizing they've grown apart from spouse over the years",
    "Being excluded from social group or work inner circle",
    "Friend's marriage appearing much happier than their own",
    "Having to care for aging parent while managing own family",
    "Colleague retiring early while they're still working",
    "Child expressing different values or life choices than expected",
    "Spouse suggesting they both need to make changes",
    "Being asked to take on additional responsibilities at work",
    "Friend achieving something they've always wanted to do",
    "Realizing they don't enjoy activities they used to love",
    "Child having problems that require professional help",
    "Spouse spending more time with friends or hobbies",
    "Being the only one in friend group without grandchildren",
    "Friend starting online dating after divorce",
    "Having argument with adult child about boundaries",
    "Colleague suggesting they're out of touch with current trends",
    "Spouse questioning whether they're still happy together",
    "Friend's child achieving success in area where theirs struggled",
    "Being asked about retirement plans when feeling unprepared",
    "Realizing they've been living according to others' expectations",
    "Friend making major life change they wish they could make",
    "Child moving back home due to financial or personal issues",
    "Spouse expressing feeling taken for granted in relationship",
    "Being the primary caregiver for multiple family members",
    "Friend losing parent and facing mortality questions",
    "Colleague being promoted to position above them",
    "Realizing they've been avoiding their own dreams and goals",
    "Friend's family seeming closer and happier than their own",
    "Being asked to make decision about parent's care facility",
    "Spouse or friend pointing out they seem unhappy or stuck",
]

SITUATIONS_OLD = [
    "Adult child suggesting they shouldn't drive anymore",
    "Doctor recommending specialist for concerning symptoms",
    "Friend being moved to assisted living facility",
    "Grandchild seeming more interested in phone than conversation",
    "Forgetting important information or repeating stories",
    "Having difficulty hearing conversation in restaurant",
    "Adult child taking over financial or medical decisions",
    "Being asked about funeral or end-of-life preferences",
    "Technology not working and needing help from younger person",
    "Feeling dizzy or unsteady when walking",
    "Adult child suggesting home modifications for safety",
    "Friend or spouse dying unexpectedly",
    "Being excluded from family decisions that affect them",
    "Physical task they used to handle easily becoming difficult",
    "Adult child expressing concern about their living situation",
    "Realizing they can't remember recent conversations clearly",
    "Being the oldest person in most social situations",
    "Doctor mentioning medication side effects or interactions",
    "Adult child seeming impatient with their pace or needs",
    "Feeling confused by new technology or procedures",
    "Being asked to consider moving closer to family",
    "Friend or peer showing signs of dementia or confusion",
    "Difficulty managing household tasks they've always done",
    "Adult child questioning their ability to live independently",
    "Feeling left out of conversations about current events",
    "Having to give up hobby or activity due to physical limitations",
    "Being treated like child by healthcare providers or others",
    "Realizing they're outliving most of their peer group",
    "Adult child suggesting they stop certain activities for safety",
    "Feeling like burden when needing help with basic tasks",
    "Being unable to participate in activities they used to enjoy",
    "Friend or family member avoiding them due to their health issues",
    "Feeling overwhelmed by medical appointments and procedures",
    "Adult child making decisions without consulting them",
    "Being patronized or talked down to by younger people",
    "Feeling scared about what will happen if health deteriorates",
    "Being asked about their wishes if they become incapacitated",
    "Feeling like their opinions and experience aren't valued",
    "Adult child suggesting changes to their living environment",
    "Struggling with grief over multiple losses in short period",
    "Feeling anxious about being alone if something happens",
    "Being told they need more supervision or assistance",
    "Feeling frustrated by physical limitations and dependency",
    "Adult child expressing worry about their judgment or decisions",
    "Being excluded from family gatherings due to health concerns",
    "Feeling like they're disappointing family with their limitations",
    "Being asked to consider hospice or palliative care options",
    "Feeling afraid about losing more independence and control",
    "Watching their world become smaller as abilities decline",
]

SIBLINGS = [
    "only child",
    "one older brother",
    "one younger sister",
    "one older sister",
    "one younger brother",
    "older brother and younger sister",
    "older sister and younger brother",
    "two older brothers",
    "two younger sisters",
    "three siblings: older sister, younger brother and sister",
]

SUPPORT_SYSTEM = [
    "mostly partner",
    "yourself",
    "mostly friends",
    "mostly family member",
    "friends and family",
    "friends and family you trust, but you don't lean on them much",
    "a few close friends who really get you",
    "professional network more than personal relationships",
    "online communities where you feel understood",
    "mentor or older colleague who gives good advice",
    "religious or spiritual community",
    "neighbors and local acquaintances you've grown close to",
    "mix of family, friends, and coworkers depending on the issue",
]

GENERAL_OUTLOOKS = [
    "mostly positive, even when stressed",
    "generally optimistic but becomes pessimistic when tired or overwhelmed",
    "neutral most of the time, leans positive when things are going well",
    "realistic with a slight negative bias, expects problems but hopes for the best",
    "alternates between very positive and very negative depending on recent events",
    "pessimistic by default but pleasantly surprised when things work out",
    "positive about big picture stuff, negative about daily inconveniences and logistics",
    "optimistic about other people's situations, pessimistic about their own prospects",
    "neutral surface demeanor but privately catastrophizes about potential problems",
    "upbeat and encouraging with others, secretly worried and negative internally",
    "positive when talking about the future, negative when reflecting on the past",
    "optimistic about work and career, pessimistic about relationships and personal life",
    "generally negative but tries to fake positivity in social situations",
    "positive during good times, spirals into deep negativity during any setbacks",
    "neutral to slightly positive, avoids both excessive optimism and pessimism",
    "cynical about systems and institutions, positive about individual people and relationships",
    "optimistic about their ability to handle problems, pessimistic about problems occurring",
    "positive outlook that gets defensive and negative when challenged or questioned",
    "neutral baseline with brief positive spikes during exciting moments or achievements",
    "negative about current circumstances, positive about potential for change and growth",
]

RELATIONSHIP_STATUS = [
    "single",
    "dating multiple people",
    "in a relationship, not married",
    "in a long-term relationship, not married",
    "married",
    "divorced",
    "widowed",
]

LIVING_SITUATIONS = {
    "single": ["alone", "with roommates"],
    "dating multiple people": ["alone with a cat", "with one roommate"],
    "in a relationship, not married": ["alone", "with one roommate"],
    "in a long-term relationship, not married": [
        "with partner",
        "with partner and pet",
    ],
    "married": ["with spouse and children", "with spouse and dog"],
    "divorced": [
        "alone",
        "with shared custody kids",
        "with their child",
        "with their children",
        "with roommate after split",
    ],
    "widowed": ["alone", "alone in family home", "with adult children who moved back"],
}

EXERCISE = [
    "inactive",
    "hardly ever",
    "barely active, occasional walks",
    "inconsistently active, goes through phases",
    "somewhat active",
    "moderately active, regular but not intense",
    "active",
    "quite active, exercise is part of routine",
    "very active",
    "highly active, trains almost daily",
    "extremely active, fitness is a major priority",
]

EXERCISE_SEVERE = [
    "inactive",
    "hardly ever",
    "barely active, occasional walks",
    "inconsistently active, goes through phases",
    "somewhat active",
    "moderately active, regular but not intense",
]

SLEEP_QUALITY = [
    "5 or fewer hours/night, non-restorative sleep that leads to feeling exhausted throughout the day",
    "5-6 hours/night, tosses and turns, takes 30+ minutes to fall asleep, groggy most mornings",
    "falls asleep instantly but wakes at 3am every night, lies awake for 1-2 hours before sleeping again",
    "6-7 hours/night but light sleep, wakes up 2-3 times, hits snooze button repeatedly",
    "6 hours/night weekdays, crashes for 10+ hours on weekends, cycles between exhausted and rested",
    "7-8 hours/night most nights, falls asleep within 15 minutes, wakes up once or twice briefly",
    "7.5-8 hours/night, sleeps deeply, wakes up naturally without alarm on weekends",
    "8-9 hours/night, prioritizes sleep above most activities, feels refreshed and energetic daily",
]

SLEEP_SEVERE = [
    "5 or fewer hours/night, non-restorative sleep that leads to feeling exhausted throughout the day",
    "5-6 hours/night, tosses and turns, takes 30+ minutes to fall asleep, groggy most mornings",
    "falls asleep instantly but wakes at 3am every night, lies awake for 1-2 hours before sleeping again",
    "6-7 hours/night but light sleep, wakes up 2-3 times, hits snooze button repeatedly",
    "6 hours/night weekdays, crashes for 10+ hours on weekends, cycles between exhausted and rested",
    "sleeps 10-12+ hours daily but still feels drained, uses sleep to escape difficult emotions or thoughts",
]

ATTITUDE_TOWARDS_MINDFULNESS = [
    "open to mindfulness practices, but not consistent with any particular method",
    "tried various wellness routines before and gave up after a few weeks",
    "believes in the benefits of mindfulness but struggles to make it a regular habit",
    "enthusiastic about personal growth in theory, procrastinates on actually doing the work",
    "skeptical of self-help approaches but willing to try if someone guides them through it",
    "sees the value in introspection but prefers talking to people over solo practices",
    "keeps starting and stopping different wellness routines, never sticks with one long-term",
    "loves the idea of mindful practices but gets overwhelmed by choosing the right approach",
    "engages in reflection sporadically during major life events, ignores it during stable periods",
    "thinks most self-improvement practices are pointless and prefers staying busy with external activities",
    "finds focusing inward uncomfortable and believes it leads to overthinking problems",
    "considers reflection-based activities a waste of time that could be spent being productive",
    "attracted to the aesthetics and community around wellness but finds the actual practices tedious",
    "turns to introspective activities only when stressed, otherwise finds them boring and forced",
    "believes mindfulness works for other people but doubts they have the patience or discipline for it",
]

REGION = ["urban", "suburban", "rural"]

EDUCATION = [
    "high school dropout, later got GED",
    "high school graduate",
    "trade school or community college graduate",
    "dropped out of college",
    "bachelor's degree",
    "master's degree",
    "professional degree (JD/MD/etc)",
    "PhD or doctorate",
]

EMPLOYMENT_STATUS = [
    "employed full time",
    "employed part time",
    "working variable hours",
    "unemployed",
    "retired",
    "working as stay-at-home parent",
]

PROFESSIONS_BY_EDUCATION = {
    "high school dropout, later got GED": [
        "Construction Worker",
        "Retail Sales Associate",
        "Food Service Worker",
        "Delivery Driver",
        "Warehouse Worker",
        "Security Guard",
        "Janitor/Custodian",
        "Farm Worker",
        "Factory Worker",
        "Home Health Aide",
        "Landscaper",
        "Cashier",
        "Kitchen Staff",
        "Taxi Driver",
        "Maintenance Worker",
    ],
    "high school graduate": [
        "Construction Worker",
        "Retail Sales Associate",
        "Food Service Worker",
        "Delivery Driver",
        "Warehouse Worker",
        "Security Guard",
        "Janitor/Custodian",
        "Farm Worker",
        "Factory Worker",
        "Home Health Aide",
        "Landscaper",
        "Cashier",
        "Kitchen Staff",
        "Taxi Driver",
        "Maintenance Worker",
    ],
    "dropped out of college": [
        "Construction Worker",
        "Retail Sales Associate",
        "Food Service Worker",
        "Delivery Driver",
        "Warehouse Worker",
        "Security Guard",
        "Janitor/Custodian",
        "Farm Worker",
        "Factory Worker",
        "Home Health Aide",
        "Landscaper",
        "Cashier",
        "Kitchen Staff",
        "Taxi Driver",
        "Maintenance Worker",
    ],
    "trade school or community college graduate": [
        "Administrative Assistant",
        "Firefighter",
        "Electrician",
        "Plumber",
        "Automotive Technician",
        "Real Estate Agent",
        "Insurance Sales Agent",
        "Bank Teller",
        "Dental Assistant",
        "Medical Assistant",
        "Court Reporter",
        "Air Traffic Controller",
        "Postal Service Worker",
        "Small Business Owner",
    ],
    "bachelor's degree": [
        "Software Developer",
        "Accountant",
        "Marketing Coordinator",
        "Elementary School Teacher",
        "Social Worker",
        "Graphic Designer",
        "Financial Analyst",
        "Human Resources Specialist",
        "Sales Manager",
        "Project Manager",
        "Journalist",
        "Civil Engineer",
        "Registered Nurse",
        "Business Analyst",
        "Web Developer",
    ],
    "master's degree": [
        "High School Principal",
        "Clinical Psychologist",
        "MBA Manager",
        "Software Architect",
        "Physical Therapist",
        "Librarian",
        "College Professor",
        "Data Scientist",
        "Urban Planner",
        "School Counselor",
        "Speech-Language Pathologist",
        "Occupational Therapist",
        "Environmental Scientist",
        "Market Research Analyst",
        "Operations Research Analyst",
    ],
    "professional degree (JD/MD/etc)": [
        "Physician",
        "Surgeon",
        "Lawyer",
        "Judge",
        "Dentist",
        "Veterinarian",
        "Pharmacist",
        "Optometrist",
        "Psychiatrist",
        "District Attorney",
        "Corporate Counsel",
        "Anesthesiologist",
        "Radiologist",
        "Patent Attorney",
        "Chief Medical Officer",
    ],
    "PhD or doctorate": [
        "University Professor",
        "Research Scientist",
        "Clinical Psychologist",
        "Epidemiologist",
        "Biostatistician",
        "Pharmaceutical Researcher",
        "Astrophysicist",
        "Economics Professor",
        "Policy Analyst",
        "Museum Curator",
        "Think Tank Researcher",
        "Government Research Director",
        "Medical Research Director",
        "Data Science Director",
        "Academic Dean",
    ],
}

FINANCIAL_SITUATIONS = {
    "high school dropout, later got GED": [
        "barely scraping by, paycheck to paycheck every month",
        "paycheck to paycheck but can handle small unexpected costs",
        "tight budget with some savings, worries about major expenses",
        "manages monthly expenses but struggles to build meaningful savings",
        "comfortable month-to-month but one big expense away from stress",
    ],
    "high school graduate": [
        "barely scraping by, paycheck to paycheck every month",
        "paycheck to paycheck but can handle small unexpected costs",
        "tight budget with some savings, worries about major expenses",
        "manages monthly expenses but struggles to build meaningful savings",
        "comfortable month-to-month but one big expense away from stress",
    ],
    "dropped out of college": [
        "barely scraping by, paycheck to paycheck every month",
        "paycheck to paycheck but can handle small unexpected costs",
        "tight budget with some savings, worries about major expenses",
        "manages monthly expenses but struggles to build meaningful savings",
        "comfortable month-to-month but one big expense away from stress",
    ],
    "trade school or community college graduate": [
        "barely scraping by, paycheck to paycheck every month",
        "paycheck to paycheck but can handle small unexpected costs",
        "tight budget with some savings, worries about major expenses",
        "manages monthly expenses but struggles to build meaningful savings",
        "comfortable month-to-month but one big expense away from stress",
    ],
    "bachelor's degree": [
        "tight budget with some savings, worries about major expenses",
        "manages monthly expenses but struggles to build meaningful savings",
        "comfortable month-to-month but one big expense away from stress",
        "comfortable income, but conscious about budgeting",
        "financially secure with investments, plans major purchases carefully",
        "high income but lifestyle inflation keeps savings modest",
    ],
    "master's degree": [
        "tight budget with some savings, worries about major expenses",
        "manages monthly expenses but struggles to build meaningful savings",
        "comfortable month-to-month but one big expense away from stress",
        "comfortable income, but conscious about budgeting",
        "financially secure with investments, plans major purchases carefully",
        "very comfortable, money stress is rare and usually investment-related",
        "high income but lifestyle inflation keeps savings modest",
        "wealthy but still budget-conscious from past financial struggles",
    ],
    "professional degree (JD/MD/etc)": [
        "tight budget with some savings, worries about major expenses",
        "comfortable income, but conscious about budgeting",
        "good income with solid emergency fund, rarely worries about money",
        "financially secure with investments, plans major purchases carefully",
        "very comfortable, money stress is rare and usually investment-related",
        "high income but lifestyle inflation keeps savings modest",
        "wealthy but still budget-conscious from past financial struggles",
        "financially independent, money decisions based on values not necessity",
        "generational wealth, never experienced financial stress personally",
    ],
    "PhD or doctorate": [
        "tight budget with some savings, worries about major expenses",
        "manages monthly expenses but struggles to build meaningful savings",
        "comfortable income, but conscious about budgeting",
        "good income with solid emergency fund, rarely worries about money",
        "financially secure with investments, plans major purchases carefully",
        "very comfortable, money stress is rare and usually investment-related",
        "wealthy but still budget-conscious from past financial struggles",
        "financially independent, money decisions based on values not necessity",
    ],
}

DEPRESSIVE_SYMPTOMS = [
    "minimal to no depressive symptoms",
    "mild depressive symptoms",
    "moderate depressive symptoms",
    "severe depressive symptoms",
]

ANXIOUS_SYMPTOMS = [
    "minimal to no anxious symptoms",
    "mild anxious symptoms",
    "moderate anxious symptoms",
    "severe anxious symptoms",
]

PROGRAM_GOALS = [
    "Reduce feelings of anxiety",
    "Reduce feelings of depression",
    "Become better at managing emotions",
    "Improve communication with others",
    "Build better habits to strengthen mental health",
    "Cope with a current major life transition",
    "Increase self-esteem",
    "Need help setting up goals",
    "Learn how to better express feelings",
    "Increase confidence in making decisions",
    "Manage intrusive thoughts",
    "Regain a sense of purpose",
]

# PROGRAM_GOALS_VERSION_DICT = {"v0_1": PROGRAM_GOALS}


def sample_one_profile(seed: int) -> list[dict[str, Union[str, int]]]:
    random.seed(seed)
    depressive_symptoms = random.choice(DEPRESSIVE_SYMPTOMS)
    anxious_symptoms = random.choice(ANXIOUS_SYMPTOMS)
    sex = random.choice(SEX)
    # prevalences taken from https://en.wikipedia.org/wiki/Heterosexuality, https://williamsinstitute.law.ucla.edu/publications/trans-adults-united-states/ , https://www.pewresearch.org/short-reads/2022/06/07/about-5-of-young-adults-in-the-u-s-say-their-gender-is-different-from-their-sex-assigned-at-birth/ (under revision)
    random.seed(seed + 1)
    sexual_orientation = (
        random.choice(SEXUAL_ORIENTATION[1:])
        if random.random() < 0.20
        else SEXUAL_ORIENTATION[0]
    )
    random.seed(seed)
    sex_gender_identity = "different" if random.random() < 0.02 else "same"
    if sex_gender_identity == "same":
        gender_identity = sex
    else:
        gender_identities = [g for g in GENDER_IDENTITY if g != sex]
        gender_identity = random.choice(gender_identities)
    name = random.choice(NAMES_BY_GENDER[gender_identity])
    age = random.choice(AGES)
    race = random.choice(RACES)
    conversation_style = random.choice(CONVERSATION_STYLES)
    thought_process = random.choice(THOUGHT_PROCESSES)
    mood = random.choice(MOODS)
    siblings = random.choice(SIBLINGS)
    general_outlook = random.choice(GENERAL_OUTLOOKS)
    if (
        depressive_symptoms == "severe depressive symptoms"
        or anxious_symptoms == "severe anxious symptoms"
    ):
        exercise = random.choice(EXERCISE_SEVERE)
        sleep_quality = random.choice(SLEEP_SEVERE)
    else:
        exercise = random.choice(EXERCISE)
        sleep_quality = random.choice(SLEEP_QUALITY)
    attitude_towards_mindfulness = random.choice(ATTITUDE_TOWARDS_MINDFULNESS)
    region = random.choice(REGION)
    if age < 36:
        if age < 21:
            education = random.choice(EDUCATION[:2])
        elif age < 23:
            education = random.choice(EDUCATION[:3])
        elif age < 27:
            education = random.choice(EDUCATION[:4])
        elif age < 30:
            education = random.choice(EDUCATION[:5])
        else:
            education = random.choice(EDUCATION)
    if age <= 27:
        financial_situation = random.choice(FINANCIAL_SITUATIONS[education][:2])
        relationship_status = random.choice(
            RELATIONSHIP_STATUS[:3]
        )  # disregard long-term relationship, married, divorced and widowed
    elif age <= 33:
        financial_situation = random.choice(FINANCIAL_SITUATIONS[education][:-2])
        relationship_status = random.choice(
            RELATIONSHIP_STATUS[:4]
        )  # disregard married, divorced and widowed
    elif age > 33 and age < 50:
        education = random.choice(EDUCATION)
        financial_situation = random.choice(FINANCIAL_SITUATIONS[education])
        relationship_status = random.choice(
            RELATIONSHIP_STATUS[:5]
        )  # disregard and widowed
    elif age >= 50:
        education = random.choice(EDUCATION)
        financial_situation = random.choice(FINANCIAL_SITUATIONS[education])
        relationship_status = random.choice(RELATIONSHIP_STATUS)
    living_situation = random.choice(LIVING_SITUATIONS[relationship_status])
    if living_situation in [
        "with spouse and children",
        "with their child",
        "with their children",
    ]:
        if age > 60:
            employment_status = random.choice(EMPLOYMENT_STATUS)
        else:
            employment_status = random.choice(
                [e for e in EMPLOYMENT_STATUS if e != "retired"]
            )
    else:
        # ignore stay-at-home parent
        if age > 60:
            employment_status = random.choice(EMPLOYMENT_STATUS[:-1])
        else:
            employment_status = random.choice(
                [
                    e
                    for e in EMPLOYMENT_STATUS
                    if e not in ["retired", "working as stay-at-home parent"]
                ]
            )
    # sample financial situation again if unemployed
    if employment_status == "unemployed" and education in [
        "bachelor's degree",
        "master's degree",
        "professional degree (JD/MD/etc)",
        "PhD or doctorate",
    ]:
        financial_situation = random.choice(
            [
                "tight budget with some savings, worries about major expenses",
                "manages monthly expenses but struggles to build meaningful savings",
                "financially secure with investments, plans major purchases carefully",
            ]
        )
    profession = random.choice(PROFESSIONS_BY_EDUCATION[education])
    support_system = (
        random.choice(SUPPORT_SYSTEM)
        if relationship_status
        not in ["single", "divorced", "widowed", "dating multiple people"]
        else random.choice(SUPPORT_SYSTEM[1:])
    )
    if employment_status in [
        "unemployed",
        "working as stay-at-home parent",
    ]:
        profession = "none"

    if "minimal" in anxious_symptoms and "minimal" in depressive_symptoms:
        program_goal = random.choice(PROGRAM_GOALS[2:])
    elif "severe" in anxious_symptoms and "severe" in depressive_symptoms:
        program_goal = random.choice(PROGRAM_GOALS[:2])
    elif "severe" in anxious_symptoms:
        program_goal = "Reduce feelings of anxiety"
    elif "severe" in depressive_symptoms:
        program_goal = "Reduce feelings of depression"
    else:
        program_goal = random.choice(PROGRAM_GOALS)
    profile = {
        "name": name,
        "sex": sex,
        "gender_identity": gender_identity,
        "sexual_orientation": sexual_orientation,
        "age": age,
        "race": race,
        "thought_process": thought_process,
        "general_outlook": general_outlook,
        "conversation_style": conversation_style,
        "recent_mood": mood,
        "education": education,
        "profession": profession,
        "employment_status": employment_status,
        "financial_situation": financial_situation,
        "support_system": support_system,
        "siblings": siblings,
        "relationship_status": relationship_status,
        "living_situation": living_situation,
        "exercise": exercise,
        "sleep_quality": sleep_quality,
        "attitude_towards_mindfulness": attitude_towards_mindfulness,
        "region": region,
        "depressive_symptoms": depressive_symptoms,
        "anxious_symptoms": anxious_symptoms,
        "program_goal": program_goal,
    }
    return profile
