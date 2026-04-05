"""
模型配置模块
定义各任务的模型路径、参数等

支持任务:
- LowLevel: 低级特征提取 (Spectral, Prosody, Energy, Temporal, Timbre)
- SCR: Speech Content Reasoning (Whisper ASR)
- ER: Emotion Recognition (emotion2vec_plus_large)
- SAR: Speaker Attribute Recognition (Age, Gender, Tone)
  - Age: age-classification (wav2vec2-based)
  - Gender: gender-classifier (ECAPA-TDNN)
  - Tone: Audio-Reasoner (Audio Reasoner: Qwen2-based)
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from pathlib import Path
import os

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_ROOT = Path("/home/u2023112559/qix/Models/Models")

# Default device
DEFAULT_DEVICE = "cuda"


# ============================================================
# 任务模型配置字典
# ============================================================
MODEL_CONFIGS = {
    # LowLevel: 低级声学特征提取
    "LowLevel": {
        "model_name": "LowLevelFeatureExtractor",
        "device": DEFAULT_DEVICE,
        "sample_rate": 16000,
        "use_vad": True,
        "features": ["spectral", "prosody", "energy", "temporal", "timbre"],
    },

    # SCR: Speech Content Reasoning (仅ASR)
    "SCR": {
        "model_name": "whisper-medium",
        "model_path": str(MODELS_ROOT / "whisper-medium"),
        "device": DEFAULT_DEVICE,
        "language": "auto",
        "task": "transcribe",
        "sample_rate": 16000,
        "enable_reasoning": False,
    },

    # SpER: Speech Entity Recognition
    "SpER": {
        "model_name": "FunASR-NER",
        "model_id": "damo/speech_timestamp_prediction-v1-16k-offline",
        "device": DEFAULT_DEVICE,
        "sample_rate": 16000,
        "entity_types": ["PER", "LOC", "ORG", "TIME", "MONEY", "PRODUCT", "DATE", "QUANTITY"],
    },

    # SED: Sound Event Detection
    "SED": {
        "model_name": "PANNs-CNN14",
        "model_path": str(MODELS_ROOT / "panns" / "Cnn14_mAP=0.431.pth"),
        "device": DEFAULT_DEVICE,
        "sample_rate": 32000,
        "audio_length": 10.0,
        "threshold": 0.5,
        "num_classes": 527,
        # 重点关注的声学事件
        "focus_events": ["Speech", "Breathing", "Child speech, kid speaking"],
    },

    # ER: Emotion Recognition (emotion2vec_plus_large)
    "ER": {
        "model_name": "emotion2vec_plus_large",
        "model_path":  str(MODELS_ROOT / "emotion2vec_plus_large"),
        "device": DEFAULT_DEVICE,
        "sample_rate": 16000,
        # emotion2vec_plus_large 官方标签映射 (0-8)
        "emotion_map": {
            0: "angry",
            1: "disgusted",
            2: "fearful",
            3: "happy",
            4: "neutral",
            5: "other",
            6: "sad",
            7: "surprised",
            8: "unknown"
        },
        "emotion_classes": ["angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unknown"],
    },

    # SAR: Speaker Attribute Recognition
    # 整合 Age + Gender + Tone 三个子任务
    "SAR": {
        "model_name": "SARAnnotator",
        "device": DEFAULT_DEVICE,
        "sample_rate": 16000,
        "enable_age": True,
        "enable_gender": True,
        "enable_tone": True,
        "sub_configs": {
            "Age": {
                "model_path": str(MODELS_ROOT / "age-classification"),
                "sample_rate": 16000,
            },
            "Gender": {
                "model_path": str(MODELS_ROOT / "gender-classifier"),
                "sample_rate": 16000,
            },
            "Tone": {
                "model_path": str(MODELS_ROOT / "Audio-Reasoner"),
                "sample_rate": 16000,
                "max_tokens": 512,
                "temperature": 0,
                "batch_size": 1,
            }
        }
    },

    # === 独立子任务配置 (可单独调用) ===
    "Age": {
        "model_name": "AgeClassifier",
        "model_path": str(MODELS_ROOT / "age-classification"),
        "device": DEFAULT_DEVICE,
        "sample_rate": 16000,
    },

    "Gender": {
        "model_name": "GenderClassifier",
        "model_path": str(MODELS_ROOT / "gender-classifier"),
        "device": DEFAULT_DEVICE,
        "sample_rate": 16000,
    },

    "Tone": {
        "model_name": "ToneAnnotator",
        "model_path": str(MODELS_ROOT / "Audio-Reasoner"),
        "device": DEFAULT_DEVICE,
        "sample_rate": 16000,
        "max_tokens": 512,
        "temperature": 0,
        "batch_size": 1,
    }
}


# AudioSet 527类标签 (PANNs使用)
AUDIOSET_LABELS = [
    "Speech", "Child speech, kid speaking", "Conversation", "Narration, monologue",
    "Babbling", "Speech synthesizer", "Shout", "Screaming", "Whispering",
    "Laughter", "Baby laughter", "Giggle", "Snicker", "Belly laugh",
    "Chuckling, chortling", "Crying, sobbing", "Baby cry, infant cry",
    "Whimpering", "Wail, moan", "Sigh", "Singing", "Choir", "Yodeling",
    "Chant", "Mantra", "Child singing", "Synthetic singing", "Rapping",
    "Humming", "Groan", "Grunt", "Whistling", "Breathing", "Wheeze",
    "Snoring", "Gasp", "Pant", "Snort", "Cough", "Throat clearing",
    "Sneeze", "Sniff", "Run", "Shuffle", "Walk, footsteps", "Clap",
    "Finger snapping", "Hand clap", "Heart sounds, heartbeat",
    "Heart murmur", "Cheering", "Applause", "Chatter", "Crowd",
    "Hubbub, speech noise, speech babble", "Audience", "Children playing",
    "Animal", "Domestic animals, pets", "Dog", "Bark", "Yip", "Howl",
    "Bow-wow", "Growling", "Whimper (dog)", "Cat", "Purr", "Meow",
    "Hiss", "Caterwaul", "Mew", "Pigeon, dove", "Coo", "Crow",
    "Caw", "Owl", "Hoot", "Bird vocalization, bird call, bird song",
    "Chirp, tweet", "Squawk", "Bird flap", "Chicken, rooster", "Cluck",
    "Crowing, cock-a-doodle-doo", "Turkey", "Gobble", "Duck", "Quack",
    "Goose", "Honk", "Fly, housefly", "Buzz", "Mosquito", "Cricket",
    "Frog", "Croak", "Pig", "Grunt", "Squeal", "Horse", "Neigh, whinny",
    "Cow", "Moo", "Sheep", "Bleat", "Goat", "Bleat", "Elephant",
    "Trumpet", "Lion", "Roar", "Bear", "Growl", "Wolf", "Howl",
    "Fox", "Deer", "Cattle, bovines", "Buffalo", "Insect", "Wild animals",
    "Rodents, rats, mice", "Mouse", "Paddle steamer, paddle wheeler",
    "Steam whistle", "Whip", "Percussion", "Drum kit", "Drum roll",
    "Drum", "Snare drum", "Rim shot", "Cymbal", "Hi-hat", "Bass drum",
    "Gong", "Tambourine", "Rattle (instrument)", "Tubular bells",
    "Mallet percussion", "Marimba, xylophone", "Glockenspiel",
    "Vibraphone", "Steelpan", "Orchestra", "Brass instrument",
    "French horn", "Trumpet", "Trombone", "Tuba", "Bass (instrument)",
    "Double bass", "Cello", "Violin, fiddle", "String instrument",
    "Banjo", "Mandolin", "Guitar", "Electric guitar", "Bass guitar",
    "Acoustic guitar", "Harp", "Piano", "Electric piano", "Organ",
    "Keyboard (musical)", "Synthesizer", "Sampler", "Harpsichord",
    "Wind instrument, woodwind instrument", "Flute", "Saxophone",
    "Clarinet", "Oboe", "Bassoon", "Bagpipes", "Didgeridoo",
    "Wind chime", "Accordion", "Harmonica", "Brass band", "Gamelan",
    "Choir", "Vocal music", "A capella", "Music", "Musical instrument",
    "Music for children", "Lullaby", "Samba", "Funk", "Rock music",
    "Heavy metal", "Punk rock", "Grunge", "Progressive rock",
    "Rock and roll", "Psychedelic rock", "Independent music",
    "Pop music", "Synth-pop", "Beat music", "Hip hop music",
    "Rap music", "Dance music", "Electronica", "Disco", "Techno",
    "Drum and bass", "Dubstep", "House music", "Electronic dance music",
    "Ambient music", "Trance music", "Soundtrack music", "New-age music",
    "Vocal jazz", "Jazz", "Swing music", "Bebop", "Scat singing",
    "Blues", "Rhythm and blues", "Soul music", "Country music",
    "Bluegrass", "Folk music", "Reggae", "Country", "Western music",
    "Middle Eastern music", "Indian music", "African music",
    "Latin American music", "Flamenco", "Salsa music", "Ska",
    "Polka", "Carnatic music", "Traditional music", "Gospel music",
    "Christian music", "Buddhist music", "Islamic music", "Jewish music",
    "Zen music", "Chant", "Mantra", "Church bell", "Jingle", "Jingle bell",
    "Bicycle bell", "Dinner bell", "Tuning fork", "Chime", "Wind chime",
    "Triangle (instrument)", "Alarm", "Alarm clock", "Siren", "Civil defense siren",
    "Buzzer", "Klaxon", "Fire alarm", "Smoke detector, smoke alarm",
    "Foghorn", "Whistle", "Steam whistle", "Train whistle", "Air horn",
    "Error message", "Beep, bleep", "Ping", "Ding", "Click", "Tick",
    "Tick-tock", "Cuckoo clock", "Gong", "Clock", "Bell", "Doorbell",
    "Knock", "Door", "Slam", "Screen door", "Sliding door", "Door knock",
    "Drawer open or close", "Cupboard open or close", "Cabinet open or close",
    "Cupboard", "Can opening", "Scissors", "Cutlery, silverware", "Chopping (food)",
    "Frying (food)", "Microwave oven", "Blender", "Water faucet, tap",
    "Sink (filling or washing)", "Bathtub (filling or washing)", "Hair dryer",
    "Toilet flush", "Toothbrush", "Vacuum cleaner", "Zipper (clothing)",
    "Velcro", "Washing machine", "Typing", "Typewriter", "Computer keyboard",
    "Keyboard (typing)", "Mouse click", "Printing", "Photocopier", "Printer",
    "Paper shredder", "Camera", "Single-lens reflex camera", "Sound effect",
    "Punch", "Slap", "Smash, crash", "Breaking", "Breaking glass",
    "Crushing", "Crash", "Collision", "Explosion", "Gunshot, gunfire",
    "Machine gun", "Fusillade", "Artillery fire", "Cap gun", "Fireworks",
    "Firecracker", "Burst, pop", "Eruption", "Boom", "Thunderstorm",
    "Thunder", "Lightning", "Rain", "Raindrop", "Rain on roof",
    "Dripping (water)", "Hail", "Water", "Ocean", "Waves, surf",
    "Stream", "Waterfall", "River", "Geyser", "Wind", "Wind noise",
    "Rustling leaves", "Leaf rustling", "Fire", "Crackle", "Ember",
    "Forest fire", "Burning", "Engine", "Engine starting", "Idling",
    "Accelerating, revving", "Vehicle", "Car", "Car passing by",
    "Race car, auto racing", "Stationary motor vehicle", "Motor vehicle",
    "Truck", "Bus", "Van", "Motorcycle", "Scooter", "Moped", "ATV",
    "Snowmobile", "Golf cart", "Tram, streetcar", "Train", "Train horn",
    "Train wheel squeal", "Subway, metro, underground", "Rail transport",
    "Boat, Watercraft", "Sailboat, sailing ship", "Motorboat, speedboat",
    "Ship", "Ship's bell", "Foghorn", "Propeller", "Aircraft", "Airplane",
    "Jet engine", "Helicopter", "Hot air balloon", "Rocket", "Spacecraft",
    "Parachute", "Fly (insect)", "Mosquito", "Bee, wasp, etc.", "Cricket",
    "Frog", "Snake", "Dinosaur", "Monster", "Ghost", "Alien", "Robot",
    "Machine", "Tools", "Drill", "Chainsaw", "Hacksaw", "Hammer",
    "Jackhammer", "Lawn mower", "Leaf blower", "Construction", "Demolition",
    "Sawing", "Filing (rasp)", "Sanding", "Grinding", "Squeak", "Creak",
    "Scrape", "Rub", "Roll", "Writing", "Drawing", "Painting", "Carving",
    "Sewing", "Knitting", "Spinning (textile)", "Weaving", "Telegraph",
    "Telephone", "Telephone dialing", "Telephone ring", "Telephone busy signal",
    "Modem", "Fax", "Siren", "Civil defense siren", "Emergency vehicle",
    "Police car (siren)", "Ambulance (siren)", "Fire engine, fire truck (siren)",
    "Fire truck", "Police car", "Ambulance", "Traffic", "Traffic noise",
    "Bicycle", "Skateboard", "Roller coaster", "Crowd", "Cheering",
    "Applause", "Laughter", "Giggle", "Cough", "Sneeze", "Burping, eructation",
    "Hiccup", "Finger snapping", "Hand clap", "Clapping", "Whistling",
    "Screaming", "Yell", "Shout", "Moan", "Grunt", "Sigh", "Groan",
    "Breathing", "Pant", "Gasp", "Wheeze", "Snore", "Snort", "Cough",
    "Throat clearing", "Spit", "Yawn", "Sneeze", "Nose blowing",
    "Laugh", "Giggle", "Snicker", "Chuckle", "Cackle", "Chortle",
    "Hoot", "Cackle", "Guffaw", "Titter", "Twitter", "Smile",
    "Chuckle", "Giggle", "Smirk", "Simper", "Grin", "Beam"
]
