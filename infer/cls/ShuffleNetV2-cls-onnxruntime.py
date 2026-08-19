import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple


class ShuffleNetV2_CLS:
    # ImageNet 1000 个类别（imagenet-simple-labels，与模型输出顺序一致）
    CLASS_NAMES = [
        'tench', 'goldfish', 'great white shark', 'tiger shark', 'hammerhead shark', 'electric ray', 'stingray', 'cock', 'hen', 'ostrich',
        'brambling', 'goldfinch', 'house finch', 'junco', 'indigo bunting', 'American robin', 'bulbul', 'jay', 'magpie', 'chickadee',
        'American dipper', 'kite', 'bald eagle', 'vulture', 'great grey owl', 'fire salamander', 'smooth newt', 'newt', 'spotted salamander', 'axolotl',
        'American bullfrog', 'tree frog', 'tailed frog', 'loggerhead sea turtle', 'leatherback sea turtle', 'mud turtle', 'terrapin', 'box turtle', 'banded gecko', 'green iguana',
        'Carolina anole', 'desert grassland whiptail lizard', 'agama', 'frilled-necked lizard', 'alligator lizard', 'Gila monster', 'European green lizard', 'chameleon', 'Komodo dragon', 'Nile crocodile',
        'American alligator', 'triceratops', 'worm snake', 'ring-necked snake', 'eastern hog-nosed snake', 'smooth green snake', 'kingsnake', 'garter snake', 'water snake', 'vine snake',
        'night snake', 'boa constrictor', 'African rock python', 'Indian cobra', 'green mamba', 'sea snake', 'Saharan horned viper', 'eastern diamondback rattlesnake', 'sidewinder', 'trilobite',
        'harvestman', 'scorpion', 'yellow garden spider', 'barn spider', 'European garden spider', 'southern black widow', 'tarantula', 'wolf spider', 'tick', 'centipede',
        'black grouse', 'ptarmigan', 'ruffed grouse', 'prairie grouse', 'peacock', 'quail', 'partridge', 'grey parrot', 'macaw', 'sulphur-crested cockatoo',
        'lorikeet', 'coucal', 'bee eater', 'hornbill', 'hummingbird', 'jacamar', 'toucan', 'duck', 'red-breasted merganser', 'goose',
        'black swan', 'tusker', 'echidna', 'platypus', 'wallaby', 'koala', 'wombat', 'jellyfish', 'sea anemone', 'brain coral',
        'flatworm', 'nematode', 'conch', 'snail', 'slug', 'sea slug', 'chiton', 'chambered nautilus', 'Dungeness crab', 'rock crab',
        'fiddler crab', 'red king crab', 'American lobster', 'spiny lobster', 'crayfish', 'hermit crab', 'isopod', 'white stork', 'black stork', 'spoonbill',
        'flamingo', 'little blue heron', 'great egret', 'bittern', 'crane (bird)', 'limpkin', 'common gallinule', 'American coot', 'bustard', 'ruddy turnstone',
        'dunlin', 'common redshank', 'dowitcher', 'oystercatcher', 'pelican', 'king penguin', 'albatross', 'grey whale', 'killer whale', 'dugong',
        'sea lion', 'Chihuahua', 'Japanese Chin', 'Maltese', 'Pekingese', 'Shih Tzu', 'King Charles Spaniel', 'Papillon', 'toy terrier', 'Rhodesian Ridgeback',
        'Afghan Hound', 'Basset Hound', 'Beagle', 'Bloodhound', 'Bluetick Coonhound', 'Black and Tan Coonhound', 'Treeing Walker Coonhound', 'English foxhound', 'Redbone Coonhound', 'borzoi',
        'Irish Wolfhound', 'Italian Greyhound', 'Whippet', 'Ibizan Hound', 'Norwegian Elkhound', 'Otterhound', 'Saluki', 'Scottish Deerhound', 'Weimaraner', 'Staffordshire Bull Terrier',
        'American Staffordshire Terrier', 'Bedlington Terrier', 'Border Terrier', 'Kerry Blue Terrier', 'Irish Terrier', 'Norfolk Terrier', 'Norwich Terrier', 'Yorkshire Terrier', 'Wire Fox Terrier', 'Lakeland Terrier',
        'Sealyham Terrier', 'Airedale Terrier', 'Cairn Terrier', 'Australian Terrier', 'Dandie Dinmont Terrier', 'Boston Terrier', 'Miniature Schnauzer', 'Giant Schnauzer', 'Standard Schnauzer', 'Scottish Terrier',
        'Tibetan Terrier', 'Australian Silky Terrier', 'Soft-coated Wheaten Terrier', 'West Highland White Terrier', 'Lhasa Apso', 'Flat-Coated Retriever', 'Curly-coated Retriever', 'Golden Retriever', 'Labrador Retriever', 'Chesapeake Bay Retriever',
        'German Shorthaired Pointer', 'Vizsla', 'English Setter', 'Irish Setter', 'Gordon Setter', 'Brittany Spaniel', 'Clumber Spaniel', 'English Springer Spaniel', 'Welsh Springer Spaniel', 'Cocker Spaniels',
        'Sussex Spaniel', 'Irish Water Spaniel', 'Kuvasz', 'Schipperke', 'Groenendael', 'Malinois', 'Briard', 'Australian Kelpie', 'Komondor', 'Old English Sheepdog',
        'Shetland Sheepdog', 'collie', 'Border Collie', 'Bouvier des Flandres', 'Rottweiler', 'German Shepherd Dog', 'Dobermann', 'Miniature Pinscher', 'Greater Swiss Mountain Dog', 'Bernese Mountain Dog',
        'Appenzeller Sennenhund', 'Entlebucher Sennenhund', 'Boxer', 'Bullmastiff', 'Tibetan Mastiff', 'French Bulldog', 'Great Dane', 'St. Bernard', 'husky', 'Alaskan Malamute',
        'Siberian Husky', 'Dalmatian', 'Affenpinscher', 'Basenji', 'pug', 'Leonberger', 'Newfoundland', 'Pyrenean Mountain Dog', 'Samoyed', 'Pomeranian',
        'Chow Chow', 'Keeshond', 'Griffon Bruxellois', 'Pembroke Welsh Corgi', 'Cardigan Welsh Corgi', 'Toy Poodle', 'Miniature Poodle', 'Standard Poodle', 'Mexican hairless dog', 'grey wolf',
        'Alaskan tundra wolf', 'red wolf', 'coyote', 'dingo', 'dhole', 'African wild dog', 'hyena', 'red fox', 'kit fox', 'Arctic fox',
        'grey fox', 'tabby cat', 'tiger cat', 'Persian cat', 'Siamese cat', 'Egyptian Mau', 'cougar', 'lynx', 'leopard', 'snow leopard',
        'jaguar', 'lion', 'tiger', 'cheetah', 'brown bear', 'American black bear', 'polar bear', 'sloth bear', 'mongoose', 'meerkat',
        'tiger beetle', 'ladybug', 'ground beetle', 'longhorn beetle', 'leaf beetle', 'dung beetle', 'rhinoceros beetle', 'weevil', 'fly', 'bee',
        'ant', 'grasshopper', 'cricket', 'stick insect', 'cockroach', 'mantis', 'cicada', 'leafhopper', 'lacewing', 'dragonfly',
        'damselfly', 'red admiral', 'ringlet', 'monarch butterfly', 'small white', 'sulphur butterfly', 'gossamer-winged butterfly', 'starfish', 'sea urchin', 'sea cucumber',
        'cottontail rabbit', 'hare', 'Angora rabbit', 'hamster', 'porcupine', 'fox squirrel', 'marmot', 'beaver', 'guinea pig', 'common sorrel',
        'zebra', 'pig', 'wild boar', 'warthog', 'hippopotamus', 'ox', 'water buffalo', 'bison', 'ram', 'bighorn sheep',
        'Alpine ibex', 'hartebeest', 'impala', 'gazelle', 'dromedary', 'llama', 'weasel', 'mink', 'European polecat', 'black-footed ferret',
        'otter', 'skunk', 'badger', 'armadillo', 'three-toed sloth', 'orangutan', 'gorilla', 'chimpanzee', 'gibbon', 'siamang',
        'guenon', 'patas monkey', 'baboon', 'macaque', 'langur', 'black-and-white colobus', 'proboscis monkey', 'marmoset', 'white-headed capuchin', 'howler monkey',
        'titi', "Geoffroy's spider monkey", 'common squirrel monkey', 'ring-tailed lemur', 'indri', 'Asian elephant', 'African bush elephant', 'red panda', 'giant panda', 'snoek',
        'eel', 'coho salmon', 'rock beauty', 'clownfish', 'sturgeon', 'garfish', 'lionfish', 'pufferfish', 'abacus', 'abaya',
        'academic gown', 'accordion', 'acoustic guitar', 'aircraft carrier', 'airliner', 'airship', 'altar', 'ambulance', 'amphibious vehicle', 'analog clock',
        'apiary', 'apron', 'waste container', 'assault rifle', 'backpack', 'bakery', 'balance beam', 'balloon', 'ballpoint pen', 'Band-Aid',
        'banjo', 'baluster', 'barbell', 'barber chair', 'barbershop', 'barn', 'barometer', 'barrel', 'wheelbarrow', 'baseball',
        'basketball', 'bassinet', 'bassoon', 'swimming cap', 'bath towel', 'bathtub', 'station wagon', 'lighthouse', 'beaker', 'military cap',
        'beer bottle', 'beer glass', 'bell-cot', 'bib', 'tandem bicycle', 'bikini', 'ring binder', 'binoculars', 'birdhouse', 'boathouse',
        'bobsleigh', 'bolo tie', 'poke bonnet', 'bookcase', 'bookstore', 'bottle cap', 'bow', 'bow tie', 'brass', 'bra',
        'breakwater', 'breastplate', 'broom', 'bucket', 'buckle', 'bulletproof vest', 'high-speed train', 'butcher shop', 'taxicab', 'cauldron',
        'candle', 'cannon', 'canoe', 'can opener', 'cardigan', 'car mirror', 'carousel', 'tool kit', 'carton', 'car wheel',
        'automated teller machine', 'cassette', 'cassette player', 'castle', 'catamaran', 'CD player', 'cello', 'mobile phone', 'chain', 'chain-link fence',
        'chain mail', 'chainsaw', 'chest', 'chiffonier', 'chime', 'china cabinet', 'Christmas stocking', 'church', 'movie theater', 'cleaver',
        'cliff dwelling', 'cloak', 'clogs', 'cocktail shaker', 'coffee mug', 'coffeemaker', 'coil', 'combination lock', 'computer keyboard', 'confectionery store',
        'container ship', 'convertible', 'corkscrew', 'cornet', 'cowboy boot', 'cowboy hat', 'cradle', 'crane (machine)', 'crash helmet', 'crate',
        'infant bed', 'Crock Pot', 'croquet ball', 'crutch', 'cuirass', 'dam', 'desk', 'desktop computer', 'rotary dial telephone', 'diaper',
        'digital clock', 'digital watch', 'dining table', 'dishcloth', 'dishwasher', 'disc brake', 'dock', 'dog sled', 'dome', 'doormat',
        'drilling rig', 'drum', 'drumstick', 'dumbbell', 'Dutch oven', 'electric fan', 'electric guitar', 'electric locomotive', 'entertainment center', 'envelope',
        'espresso machine', 'face powder', 'feather boa', 'filing cabinet', 'fireboat', 'fire engine', 'fire screen sheet', 'flagpole', 'flute', 'folding chair',
        'football helmet', 'forklift', 'fountain', 'fountain pen', 'four-poster bed', 'freight car', 'French horn', 'frying pan', 'fur coat', 'garbage truck',
        'gas mask', 'gas pump', 'goblet', 'go-kart', 'golf ball', 'golf cart', 'gondola', 'gong', 'gown', 'grand piano',
        'greenhouse', 'grille', 'grocery store', 'guillotine', 'barrette', 'hair spray', 'half-track', 'hammer', 'hamper', 'hair dryer',
        'hand-held computer', 'handkerchief', 'hard disk drive', 'harmonica', 'harp', 'harvester', 'hatchet', 'holster', 'home theater', 'honeycomb',
        'hook', 'hoop skirt', 'horizontal bar', 'horse-drawn vehicle', 'hourglass', 'iPod', 'clothes iron', "jack-o'-lantern", 'jeans', 'jeep',
        'T-shirt', 'jigsaw puzzle', 'pulled rickshaw', 'joystick', 'kimono', 'knee pad', 'knot', 'lab coat', 'ladle', 'lampshade',
        'laptop computer', 'lawn mower', 'lens cap', 'paper knife', 'library', 'lifeboat', 'lighter', 'limousine', 'ocean liner', 'lipstick',
        'slip-on shoe', 'lotion', 'speaker', 'loupe', 'sawmill', 'magnetic compass', 'mail bag', 'mailbox', 'tights', 'tank suit',
        'manhole cover', 'maraca', 'marimba', 'mask', 'match', 'maypole', 'maze', 'measuring cup', 'medicine chest', 'megalith',
        'microphone', 'microwave oven', 'military uniform', 'milk can', 'minibus', 'miniskirt', 'minivan', 'missile', 'mitten', 'mixing bowl',
        'mobile home', 'Model T', 'modem', 'monastery', 'monitor', 'moped', 'mortar', 'square academic cap', 'mosque', 'mosquito net',
        'scooter', 'mountain bike', 'tent', 'computer mouse', 'mousetrap', 'moving van', 'muzzle', 'nail', 'neck brace', 'necklace',
        'nipple', 'notebook computer', 'obelisk', 'oboe', 'ocarina', 'odometer', 'oil filter', 'organ', 'oscilloscope', 'overskirt',
        'bullock cart', 'oxygen mask', 'packet', 'paddle', 'paddle wheel', 'padlock', 'paintbrush', 'pajamas', 'palace', 'pan flute',
        'paper towel', 'parachute', 'parallel bars', 'park bench', 'parking meter', 'passenger car', 'patio', 'payphone', 'pedestal', 'pencil case',
        'pencil sharpener', 'perfume', 'Petri dish', 'photocopier', 'plectrum', 'Pickelhaube', 'picket fence', 'pickup truck', 'pier', 'piggy bank',
        'pill bottle', 'pillow', 'ping-pong ball', 'pinwheel', 'pirate ship', 'pitcher', 'hand plane', 'planetarium', 'plastic bag', 'plate rack',
        'plow', 'plunger', 'Polaroid camera', 'pole', 'police van', 'poncho', 'billiard table', 'soda bottle', 'pot', "potter's wheel",
        'power drill', 'prayer rug', 'printer', 'prison', 'projectile', 'projector', 'hockey puck', 'punching bag', 'purse', 'quill',
        'quilt', 'race car', 'racket', 'radiator', 'radio', 'radio telescope', 'rain barrel', 'recreational vehicle', 'reel', 'reflex camera',
        'refrigerator', 'remote control', 'restaurant', 'revolver', 'rifle', 'rocking chair', 'rotisserie', 'eraser', 'rugby ball', 'ruler',
        'running shoe', 'safe', 'safety pin', 'salt shaker', 'sandal', 'sarong', 'saxophone', 'scabbard', 'weighing scale', 'school bus',
        'schooner', 'scoreboard', 'CRT screen', 'screw', 'screwdriver', 'seat belt', 'sewing machine', 'shield', 'shoe store', 'shoji',
        'shopping basket', 'shopping cart', 'shovel', 'shower cap', 'shower curtain', 'ski', 'ski mask', 'sleeping bag', 'slide rule', 'sliding door',
        'slot machine', 'snorkel', 'snowmobile', 'snowplow', 'soap dispenser', 'soccer ball', 'sock', 'solar thermal collector', 'sombrero', 'soup bowl',
        'space bar', 'space heater', 'space shuttle', 'spatula', 'motorboat', 'spider web', 'spindle', 'sports car', 'spotlight', 'stage',
        'steam locomotive', 'through arch bridge', 'steel drum', 'stethoscope', 'scarf', 'stone wall', 'stopwatch', 'stove', 'strainer', 'tram',
        'stretcher', 'couch', 'stupa', 'submarine', 'suit', 'sundial', 'sunglass', 'sunglasses', 'sunscreen', 'suspension bridge',
        'mop', 'sweatshirt', 'swimsuit', 'swing', 'switch', 'syringe', 'table lamp', 'tank', 'tape player', 'teapot',
        'teddy bear', 'television', 'tennis ball', 'thatched roof', 'front curtain', 'thimble', 'threshing machine', 'throne', 'tile roof', 'toaster',
        'tobacco shop', 'toilet seat', 'torch', 'totem pole', 'tow truck', 'toy store', 'tractor', 'semi-trailer truck', 'tray', 'trench coat',
        'tricycle', 'trimaran', 'tripod', 'triumphal arch', 'trolleybus', 'trombone', 'tub', 'turnstile', 'typewriter keyboard', 'umbrella',
        'unicycle', 'upright piano', 'vacuum cleaner', 'vase', 'vault', 'velvet', 'vending machine', 'vestment', 'viaduct', 'violin',
        'volleyball', 'waffle iron', 'wall clock', 'wallet', 'wardrobe', 'military aircraft', 'sink', 'washing machine', 'water bottle', 'water jug',
        'water tower', 'whiskey jug', 'whistle', 'wig', 'window screen', 'window shade', 'Windsor tie', 'wine bottle', 'wing', 'wok',
        'wooden spoon', 'wool', 'split-rail fence', 'shipwreck', 'yawl', 'yurt', 'website', 'comic book', 'crossword', 'traffic sign',
        'traffic light', 'dust jacket', 'menu', 'plate', 'guacamole', 'consomme', 'hot pot', 'trifle', 'ice cream', 'ice pop',
        'baguette', 'bagel', 'pretzel', 'cheeseburger', 'hot dog', 'mashed potato', 'cabbage', 'broccoli', 'cauliflower', 'zucchini',
        'spaghetti squash', 'acorn squash', 'butternut squash', 'cucumber', 'artichoke', 'bell pepper', 'cardoon', 'mushroom', 'Granny Smith', 'strawberry',
        'orange', 'lemon', 'fig', 'pineapple', 'banana', 'jackfruit', 'custard apple', 'pomegranate', 'hay', 'carbonara',
        'chocolate syrup', 'dough', 'meatloaf', 'pizza', 'pot pie', 'burrito', 'red wine', 'espresso', 'cup', 'eggnog',
        'alp', 'bubble', 'cliff', 'coral reef', 'geyser', 'lakeshore', 'promontory', 'shoal', 'seashore', 'valley',
        'volcano', 'baseball player', 'bridegroom', 'scuba diver', 'rapeseed', 'daisy', "yellow lady's slipper", 'corn', 'acorn', 'rose hip',
        'horse chestnut seed', 'coral fungus', 'agaric', 'gyromitra', 'stinkhorn mushroom', 'earth star', 'hen-of-the-woods', 'bolete', 'ear of corn', 'toilet paper',
    ]

    # ImageNet mean/std（官方标准归一化）
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, onnx_model: str, topk: int = 5, draw_boxes: bool = False):
        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        self.session = ort.InferenceSession(onnx_model, providers=providers or available)
        self.model_inputs = self.session.get_inputs()
        input_shape = self.model_inputs[0].shape
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]

        # 由输出维度推断类别数（ImageNet 为 1000）
        self.num_classes = 1000
        out_shape = self.session.get_outputs()[0].shape
        if len(out_shape) >= 2 and isinstance(out_shape[1], int) and out_shape[1] > 0:
            self.num_classes = out_shape[1]

        self.topk = topk
        # 仅控制是否在图上绘制 Top-1 类别标签（分类模型不返回检测框）
        self.draw_boxes = draw_boxes

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        官方 ImageNet 预处理：resize 短边至 256 -> 中心裁剪到模型输入尺寸(224×224)
        -> ToTensor(/255, HWC->CHW) -> Normalize(ImageNet mean/std)。
        仿照官方 test.py 的 ToTensor + Normalize 流程。
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        scale = 256.0 / min(h, w)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        img_resized = cv2.resize(img_rgb, (new_w, new_h))
        rh, rw = img_resized.shape[:2]
        x0 = max((rw - self.input_width) // 2, 0)
        y0 = max((rh - self.input_height) // 2, 0)
        img_crop = img_resized[y0:y0 + self.input_height, x0:x0 + self.input_width]

        mean = np.array(self.IMAGENET_MEAN, dtype=np.float32)[None, :, None, None]
        std = np.array(self.IMAGENET_STD, dtype=np.float32)[None, :, None, None]
        image_data = img_crop.astype(np.float32) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))[None]
        image_data = (image_data - mean) / std
        return image_data.astype(np.float32)

    def postprocess(self, outputs: List[np.ndarray]) -> List[Tuple[int, str, float]]:
        """
        官方 test.py 对 logits 取 topk，这里先 softmax 得到概率再取 Top-k（顺序一致）。
        返回 [(class_id, class_name, confidence)]，按置信度降序。
        """
        logits = outputs[0][0]  # [num_classes]
        exp = np.exp(logits - np.max(logits))
        probs = exp / np.sum(exp)
        top_indices = np.argsort(probs)[::-1][:self.topk]
        results = [(int(i), self.CLASS_NAMES[i], float(probs[i])) for i in top_indices]
        return results

    def draw_detections(self, img: np.ndarray, results: List[Tuple[int, str, float]]) -> None:
        """在图像左上角绘制 Top-1 类别标签"""
        idx, name, conf = results[0]
        label = f"{name}: {conf:.3f}"
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (0, 0), (label_width + 20, label_height + 20),
                      (0, 0, 0), cv2.FILLED)
        cv2.putText(img, label, (10, label_height + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)

    def run(self, img: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, str, float]]]:
        """
        执行推理，返回 (绘制后的图像, Top-k 预测结果)
        """
        img_copy = img.copy()
        img_data = self.preprocess(img)
        outputs = self.session.run(None, {self.model_inputs[0].name: img_data})
        results = self.postprocess(outputs)
        if self.draw_boxes:
            self.draw_detections(img_copy, results)
        return img_copy, results


if __name__ == "__main__":
    from pathlib import Path

    script_dir = Path(__file__).resolve().parent
    target_file = script_dir.parent.parent
    model = target_file / Path("models/cls/ShuffleNetV2-cls-onnxruntime.onnx")
    img = target_file / Path("assets/fish.png")
    topk = 5

    # 实例化时设置 draw_boxes=True 可绘制
    classification = ShuffleNetV2_CLS(model, topk=topk, draw_boxes=True)
    output_image, results = classification.run(cv2.imread(str(img)))
    print(f"Top-{len(results)} 预测：")
    for i, (idx, name, conf) in enumerate(results):
        print(f"  {i + 1}. {name} (class_id={idx}): {conf:.3f}")

    cv2.imshow("Output", output_image)
    cv2.waitKey(0)
