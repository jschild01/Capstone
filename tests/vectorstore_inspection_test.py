import os
import sys
from typing import Dict, Any, List, Tuple
import json
from datetime import datetime
import deeplake
from langchain.schema import Document
from langchain_community.vectorstores import DeepLake
import shutil

# Add the src directory to the Python path
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)

from component.rag_retriever_deeplake import RAGRetriever
from component.logging_config import setup_logging

# Define paths
DATA_DIR = "/home/ubuntu/Capstone/data"
LOG_DIR = os.path.join(project_root, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

logger = setup_logging()


def get_directory_size(path: str) -> float:
    """Calculate total size of a directory in MB."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / 1024 / 1024  # Convert to MB


def log_test_results(test_name: str, results: Dict[str, Any]):
    """Log test results to a JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"vectorstore_test_{timestamp}.json")

    log_data = {
        "test_name": test_name,
        "timestamp": timestamp,
        "results": results
    }

    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)

    logger.info(f"Test results logged to: {log_file}")


def inspect_dataset(dataset_path: str) -> Dict[str, Any]:
    """Inspect the DeepLake dataset contents in detail."""
    try:
        logger.info(f"\nInspecting dataset at: {dataset_path}")
        total_size = get_directory_size(dataset_path)
        logger.info(f"Total dataset size on disk: {total_size:.2f} MB")

        ds = deeplake.load(dataset_path)

        info = {
            "tensors": {},
            "total_samples": 0,
            "storage_info": {},
            "directory_structure": {}
        }

        # Get storage details
        storage_path = os.path.join(dataset_path, 'storage')
        if os.path.exists(storage_path):
            storage_files = os.listdir(storage_path)
            logger.info("\nStorage contents:")
            for file in storage_files:
                file_path = os.path.join(storage_path, file)
                size = os.path.getsize(file_path)
                logger.info(f"- {file}: {size / 1024 / 1024:.2f} MB")
                info["storage_info"][file] = size

        # Inspect tensors
        logger.info("\nTensor Details:")
        for tensor_name in ds.tensors:
            tensor = ds.tensors[tensor_name]
            shape = tensor.shape
            dtype = tensor.dtype if hasattr(tensor, 'dtype') else 'unknown'
            storage_size = 0

            # Try to get tensor storage size
            tensor_path = os.path.join(storage_path, tensor_name) if os.path.exists(storage_path) else None
            if tensor_path and os.path.exists(tensor_path):
                storage_size = get_directory_size(tensor_path)

            info["tensors"][tensor_name] = {
                "shape": shape,
                "dtype": str(dtype),
                "storage_size_mb": storage_size,
                "samples": []
            }

            logger.info(f"\nTensor: {tensor_name}")
            logger.info(f"Shape: {shape}")
            logger.info(f"Type: {dtype}")
            logger.info(f"Storage size: {storage_size:.2f} MB")

            if info["total_samples"] == 0 and len(shape) > 0:
                info["total_samples"] = shape[0]

            # Sample first few elements if tensor is not empty
            if shape and shape[0] > 0:
                try:
                    sample_size = min(2, shape[0])
                    logger.info(f"First {sample_size} elements:")
                    for i in range(sample_size):
                        sample = tensor[i].numpy()
                        if isinstance(sample, bytes):
                            sample = sample.decode('utf-8')
                        if isinstance(sample, (list, tuple)) and len(sample) > 100:
                            sample = f"{str(sample[:100])}... (truncated)"
                        logger.info(f"Element {i}: {sample}")
                        info["tensors"][tensor_name]["samples"].append(str(sample))
                except Exception as e:
                    logger.error(f"Error sampling tensor {tensor_name}: {e}")

        # Check dataset directory structure
        logger.info("\nDataset Directory Structure:")
        for root, dirs, files in os.walk(dataset_path):
            level = root.replace(dataset_path, '').count(os.sep)
            indent = ' ' * 4 * level
            subpath = root.replace(dataset_path, '').lstrip(os.sep)
            info["directory_structure"][subpath] = {"files": {}}

            logger.info(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                size = os.path.getsize(os.path.join(root, f))
                logger.info(f"{subindent}{f}: {size / 1024 / 1024:.2f} MB")
                info["directory_structure"][subpath]["files"][f] = size

        return info

    except Exception as e:
        logger.error(f"Error inspecting dataset: {e}")
        return {"error": str(e)}


def test_sample_queries(retriever: RAGRetriever) -> List[Tuple[str, int, bool, bool]]:
    """Run a series of test queries and return results with document matching.

    Returns:
        List of tuples containing (query, num_results, content_match, doc_match)
    """
    test_queries = [
        ("What did Johnny Garrison say was the exact date he started his business?", "afs21189a_en.txt",
         "March 19, 1939"),

        ("Before owning his own store, where did Johnny Garrison say he worked first?", "afs21189a_en.txt",
         "Rogers then, now Colonial"),

        ("What was Slop Jackson's memorable greeting according to the interview?", "afs21189a_en.txt",
         "Fine, and dandy myself, and I hope you're all right"),

        ("What ingredients did the Vietnamese interviewee say go into pho?", "mb_r019_01_en.txt",
         "Noodles, they put some soup in there, some beef, some vegetables"),

        ("When did Johnny Garrison purchase his store from Pittman Grocery Company?", "afs21189a_en.txt",
         "October 20, 1945"),

        ("What was the profit-sharing arrangement when Johnny Garrison managed the store for Pittman?",
         "afs21189a_en.txt", "I got 50 cents"),

        ("Why did the Vietnamese interviewee say their parents couldn't come to America?", "mb_r019_01_en.txt",
         "English, you know, second language"),

        ("What was Brown's characteristic greeting according to the interview?", "afs21189a_en.txt",
         "Good to be here. So many done been here and gone on"),

        ("Complete this sentence: 'I managed it for Pittman Grocery Company in a most unusual and'", "afs21189a_en.txt",
         "unique way"),

        (
        "Complete this sentence: 'Under the same circumstances, when you met Brown on the street, he stopped and his first word was'",
        "afs21189a_en.txt", "Good to be here"),

        (
        "Complete this sentence: 'When I see this old mill house in the bottom of the deepest part of Crystal Lake, where there's supposed to be'",
        "afs21189a_en.txt", "no bottom to it"),

        ("Complete this sentence: 'Because when the old people live in my country, they are very'", "mb_r019_01_en.txt",
         "friendship"),

        ("Complete this sentence: 'First we eat, then we do'", "dt_r018_01_en.txt", "business"),

        ("Complete this sentence: 'They're becoming more prevalent in this area than they'", "afs21189a_en.txt",
         "used to be"),

        ("Complete this sentence: 'One morning he went down to do some corn growing, and found his mill had'",
         "afs21189a_en.txt", "sunk into Crystal Lake"),

        ("Complete this sentence: 'It's the only stream that I know of in this county that runs'", "afs21218a_en.txt",
         "north"),

        ("Complete this sentence: 'They obtained my merchandise at'", "afs21189a_en.txt", "broker's cost"),

        ("Complete this sentence: 'They just want to go to their friend. They don't need to make'", "mb_r019_01_en.txt",
         "appointment"),
        ("Complete this sentence: 'The mules are not hungry. They're lively and'", "sr22a_en.txt", "gay"),
        ("Complete this sentence: 'Take a trip on the canal if you want to have'", "sr28a_en.txt or sr13a_en.txt",
         "fun"),
        (
        "What is the name of the female character mentioned in the song that begins 'In Scarlett town where I was born'?",
        "sr02b_en.txt", "Barbrae Allen"),
        ("According to the transcript, what is Captain Pearl R. Nye's favorite ballad?", "sr28a_en.txt",
         "Barbara Allen"),
        ("Complete this phrase from the gospel train song: 'The gospel train is'", "sr26a_en.txt", "night"),
        ("In the song 'Barbara Allen,' where was Barbara Allen from?", "sr02b_en.txt", "Scarlett town"),
        ("In the song 'Lord Lovele,' how long was Lord Lovele gone before returning?", "sr08a_en.txt",
         "A year or two or three at most"),
        ("What instrument does Captain Nye mention loving?", "sr22a_en.txt", "old fiddled mouth organ banjo"),
        ("In the song about pumping out Lake Erie, what will be on the moon when they're done?", "sr27b_en.txt",
         "whiskers"),
        ("Complete this line from a song: 'We land this war down by the'", "sr05a_en.txt", "river"),
        ("What does the singer say they won't do in the song 'I Won't Marry At All'?", "sr01b_en.txt",
         "Marry/Mary at all"),
        ("What does the song say will 'outshine the sun'?", "sr17b_en.txt", "We'll/not"),
        ("In the 'Dying Cowboy' song, where was the cowboy born?", "sr20b_en.txt", "Boston"),
        ("When singing about the mountains, in what should I burn in?", "sr10a_en.txt", "the trees/trees"),
        ('What is the farmer doing being sung about?', 'sr10a_en.txt', 'planting corn and beans'),
        ('Who killed Ka-Raban?', 'sr23b_en.txt', 'I killed Ka-Raban'),
        ('Who is Kevin Perlarni?', 'sr28a_en.txt', 'the last of the Ohio Canal Captains'),
        ("How many boys and girls were in the Captain's family?", 'sr28a_en.txt', '18 of us 11 boys and seven girls'),
        ('Why did Mr. Lomax travel to acrohone?', 'sr28a_en.txt',
         'In the autumn of 1936, I received a letter from a lady reporter in the acrohone of how, saying that a wonderful ballot singer lived in that town. I packed my recording equipment in my car and arrived in acrohone the following week.'),
        ('How much money do you play for?', 'sr27a_en.txt', 'I play for an nickel I play for a dime.'),
        ('Who has dancing eyes?', 'sr27a_en.txt', 'And dear mother nature with kind dancing eyes.'),
        ('Where does the gambling happen?', 'sr27a_en.txt or sr22a_en.txt', 'I gamble in Cleveland. (also see below)'),
        ('If the gambling is not in Cleveland, then where does it take place?', 'sr21a_en.txt',
         "I gamble down in Washington. I gamble down in Spain. I'm going down in Georgia to gamble my last game."),
        ('Who is the doctor of Danielle Loughano?', 'sr21a_en.txt',
         "I am Rowing Gamble...I'm the doctor of Danielle Loughano."),
        ('Rowing Gamble, the doctor of Danielle Loughano, met a girl. Describe her appearance. ', 'sr21a_en.txt',
         'Her eyes were like twos sparkling diamonds. As the stars of a clear frosty night. Her cheeks were too blooming roses. And a teeth of the ivory so white. She was emboiled a goddess of freedom. And green ones the mantle she wore.'),
        ("What's the gospel train?", 'sr26a_en.txt', 'The gospel train is night.'),
        ('What is coming out tonight?', 'sr25b_en.txt', 'Oh yellow but yellow is actually coming out tonight.'),
        ('Tell me about the girl with beautiful face and smile.', 'sr25b_en.txt',
         'There lives a girl literally, Love with beautiful face and smile. Her cheeks are like still red. Red rose and eyes are lovely brown. Her hair is long and beauty follow me.'),
        ('When are you meeting Mary?', 'sr25b_en.txt',
         "Pretty little Mary, my known boat's very. Oh my turtle dove. I'll meet you when the sun goes down."),
        ('Who are you meeting at sun down?', 'sr25b_en.txt',
         "Pretty little Mary is a keeper of a dairy and so. How I love. I'll meet you when the sun goes down."), (
        'Why were they praying on the ship?', 'sr25a_en.txt',
        "In a storm of sea. To the fearful thing in winter to be scattered by the blast. And to hear the rallying trumpet thunder cut away the mast. We were crowded in the cabin not a soul would dare to sleep. While the hungry sea was roaring and the storm was on the deep. Our ship was like a feather while the stouter's tail is breath. And the angry sea was roaring as the breaker threatened there. So we hovered there in silence each one busy in his prayers."),
        ('What did the little girl take while on the ship?', 'sr25a_en.txt',
         'Then his little daughter whispered as she took his icy hands.'), (
        'Why must you use your common sense?', 'sr24b_en.txt',
        'Experience I know is best so you must use your common sense.'), ('how much does bread cost?', 'sr24b_en.txt',
                                                                          "Just one penny please oh mister, just one penny to buy bread. Just one penny's all I ask you, just one penny I know more."),
        ('Why does mother need or want bread?', 'sr24b_en.txt',
         "Mother weeps I know she's failing and may die for one to bread. Just one penny's all I ask you, just one penny I'll be saying. Mother sick and much I worry that till die for one to bread."),
        ('Who should I stay away from and give room?', 'sr24a_en.txt',
         'So boys keep away from the girls I see and give them lots of room.'),
        ('where should I big a big hole?', 'sr23a_en.txt', 'And dig a big hole in the center'),
        ('how many good fellows are there?', 'sr23a_en.txt', 'Then let those six trusty good fellows'), (
        'For what reason did I show the world I did for?', 'sr22b_en.txt',
        'So take my grave both wide and deep, place a marble stone, and not my head and feet, and on my breast, that turtle, though, to show to the world that I died for love.'),
        ("I eat when I'm hungry. When do I drink?", 'sr22a_en.txt', "I eat when I'm hungry and drink when my dry."),
        ('What do I like better than a gay time?', 'sr22a_en.txt', 'I like a gay time but I love just one girl.'), (
        'Why does Rowing Gamble put money down?', 'sr21a_en.txt',
        'Rowing Gamble. I am Rowing Gamble. I am going down in town. Whenever I meet a deck of cards I lay my money down'),
        ('Describe how the young cowboy was dressed and looked.', 'sr20b_en.txt',
         'I saw a young cowboy. All dressed in white linen. With cold black eyes and waving black hair.'),
        ('What did the boy do in town?', 'sr20a_en.txt', 'but all the boys in our town went out to toss their ball'), (
        'What did the rich lady from London call herself?', 'sr19b_en.txt',
        'There was a rich lady from London she came. She called herself silly. Pretty silly by name.'), (
        'Why must Sally suffer?', 'sr19b_en.txt',
        "Oh, Sally, oh, Sally. Oh, Sally said he, oh don't you remember. How you slighted me. You treated me like me, my love you discord. So now you must suffer for the past you have done."),
        ('Who grew sick and denied treatment from the doctor', 'sr19b_en.txt',
         "Pretty Sally grew sick and she pitchy would die. She tangled words and lovin' herself she accused. So sent for the doctor she once had refused. Oh, am I that doctor? Who skill you would try? Or am I the young man? You once did deny? Yes you were the doctor can cure, can cure."),
        ("What is Highland White's title?", 'sr18b_en.txt',
         "I'll spoke say hard to you, Highland White, I'll go my chief, I'm ready."), (
        "What word or phrase is used to rhyme with this line: 'should this starsteps discover'", 'sr18b_en.txt',
        'when they had slain her lover'),
        ('where does the damsel that dwell?', 'sr18a_en.txt', 'In London, sweet city, a fair damsel that dwells.'), (
        "What was the woman's name who was courted by a sailor?", 'sr18a_en.txt',
        'She was courted by sailor for two bayous brine, and him two is trade with a shippipe and a shippipe. He says, my Miss Mary, if you will agree, if you will consent, go along with me.'),
        ('How did the blood flow from her body?', 'sr18a_en.txt',
         'And the blood from her body, like a fountain did flow.'), (
        'What were the others doing on the train?', 'sr18a_en.txt',
        'Some were reading, some were drinking, some were sleeping, some were laughing and some they cry.'), (
        'What did she say half crying?', 'sr17b_en.txt',
        "Oh, don't leave me now, this she said half crying. Be manly and brave."), (
        'What happened on the banks of silly?', 'sr17a_en.txt',
        "we'll both sport together on the banks of Silly…How happy we will be, and we'll both sport together, on the banks of Silly."),
        ('How was the weather on the eighth day of March?', 'sr14b_en.txt',
         'On the eight day of March about ten in the morning, The sky it was cloudless and bright shown the sun.'), (
        "what phrase was used to rhyme with 'rebel flag flew'?", 'sr14b_en.txt',
        'The turmen took conquer the Comberland crew'), (
        'What is described as a "dangerous, terrific power"?', 'sr02a_en.txt',
        "Her gun's a dangerous, a terrific power that savages,"), (
        'What ultimately happened to the British League?', 'sr02a_en.txt',
        'The whole British League was captured completely.'), (
        "What unfortunate news is delivered to Bird's parents in the letter?", 'sr03a_en.txt',
        "The letter to Bird's parents contained distressing news that Bird was to suffer for deserting from the brig 'Naiga,' indicating that he was ordered to die."),
        ('What happens to the servant who overhears the conversation in the song?', 'sr04a_en.txt',
         "One of Lord benefits servants then, who overheard it all. He randomly came to the river side, and he lit the thin and swam.He swam till he came to the other side, and then he lit out and ran. He randomly came to King George's gate, he rattled and rattled and rang."),
        ('What behavior change does the song describe about the man after he was married?', 'sr03b_en.txt',
         "Before we were married, he's models of kind, and there everyone spoke of him well. But now he goes out and my heart's full of doubts and nothing to me will he tell."),
        ('How did the woman in the song respond when her husband wanted to fight?', 'sr03b_en.txt',
         'My nerves give way out and I cry. And in that he delights and not wants to fight, but please him so often I try. But one day I accepted his challenge, So grand with a flat iron I knocked him quite cold.'),
        ('What is repeatedly mentioned as being kept open in the song?', 'sr05a_en.txt',
         'The song repeatedly mentions keeping the "golden gates" wide open.'), (
        'In the song about the devil and a river side, whos face was seen in a place?', 'sr05a_en.txt',
        "For I'm going to a place where I've seen my Savior's face."),
        ('According to the song, is the Bible true or false?', 'sr05a_en.txt', "The Bible's true."), (
        'What is the name of the hill being climbed towards the golden gates?', 'sr05a_en.txt',
        "But keep those golden gays wide open. For I am climbing Zion's hill. So keep those gays a jar."), (
        'What phrase is repeated in the song about the people all dressed in white?', 'sr11b_en.txt',
        "Mary don't you read"),
        ('What does Sally do while singing?', 'sr11b_en.txt', 'Sally often chuckles while they sing this little song'),
        ('How does the lady try to defend Riley?', 'sr10b_en.txt',
         'The lady with the tear began and the sweet fly-g. The fall was done a while is a blame lies with me. I forced him to leave this place and come along with me. I loved him out of Malaysia which brought back destiny.'),
        ("What consequence does Willy Riley face for eloping with the young woman against her father's wishes?",
         'sr10b_en.txt',
         "Willy Riley faces imprisonment and the threat of severe punishment for eloping with the young woman against her father's wishes."),
        ('What valuable items does the young woman mention taking with her when she eloped with Willy Riley?',
         'sr10b_en.txt',
         'The young woman mentions taking diamonds, rings, a watch, silver, and other precious things with her when she eloped with Willy Riley, amounting to a value of more than 500 pounds.'),
        ("What is Mr. Soblet's plight in the song?", 'sr11a_en.txt',
         'Mr. Soblet appears to be in a well, expressing distress or an "awful yell" as described in the song.'),
        ('How did Soblet fall into a well?', 'sr11a_en.txt', 'I lost my balance and my fell-whoo.'), (
        'Who did Mr. Hunter come after?', 'sr11a_en.txt',
        'Then Mr. Rattles-Nake-whoo. Then Mr. Rattles-Nake-whoo said shut the door. I love you all whoo. Soon after Mr. Hunter came to whoo. Soon after Mr. Hunter came, He raised his gun with deadly aewo.'),
        ('What challenges are described as being faced on the "knallet" as per the song?', 'sr14a_en.txt',
         'The song describes facing challenges with robbers, skaters, bed bugs, and other pests such as roaches and crickets, all adding to the struggles and lively experiences on the "knallet."'),
        ('Why did the man work and cry after learning the truth?', 'sr14a_en.txt',
         'Not knowing she was dead and buried. With thoughts of her he was occupied. When he arrived to her home he hastened. The truth he learned he worked and cried. The sunshine of his life had vanished.'),
        ('What action does the song frequently urge to be taken regarding the baby?', 'sr07b_en.txt',
         'The song frequently urges to stop the noise and keep the baby still, emphasizing the need for quiet in order to calm or soothe the baby.'),
        ('What did the woman dream about one night?', 'sr07a_en.txt',
         'She had a dream that night that her lover was killed and she saw the blood running.'), (
        'How does the woman predict her father will die?', 'sr07a_en.txt',
        "And he told her she told him what would happen. He'd die public show."),
        ('Who was the third that came in?', 'sr05b_en.txt', "The third came in was Lord Jennifer's wife"), (
        'Who wears a ring on their finger?', 'sr05b_en.txt',
        'For I know by the ring that you wear on your finger, you are Lord Bennett for twice.'), (
        'How was Lord Bennett identified?', 'sr05b_en.txt',
        'For I know by the ring that you wear on your finger, you are Lord Bennett for twice.'), (
        'What bird is the clever tailor compared to?', 'sr12a_en.txt',
        'Therely the tailor like a hawk in the stall, Tuberralli Tally,'), (
        "Where is Jenny's piece of bread?", 'sr12b_nn.txt or sr12b_nn_en_translation.txt',
        "And scorn she bows and starts to row my gentle let go Jenny. There's a piece of bread upon the shelf, see it there."),
        ('Which is better, a trip to Omnick Great Lakes or a trip on grail boats?', 'sr13a_en.txt',
         'You may talk of your pleasure trip to Omnick Great Lakes but a trip on these grail boats to bet takes the cake.'),
        ('What caught the old boy and caused him to eat right for a long time?', 'sr13a_en.txt',
         "Her hip nodded guys (hipnotic gaze?) and that wonderful smile would catch the old boy and he'd eat right for miles."),
        ("What was Mrs. Dalligim's previous name?", 'sr21b_en.txt',
         'She changed her name from lovely dime to Mrs. Dalligim of caroline'), (
        'Where did the man go after dressing himself from top to toe?', 'sr21b_en.txt',
        'I dressed myself from top to toe and down to diner I did go'), (
        'Is the house cruffender nice or mean, and why did he get left?', 'sr01a_en.txt',
        "For I have married a house-cruffender, and I think he's a nice young man, and I think he's a nice young man. If you will leave your house-cruffender, and go along with me,"),
        ('How long was the woman at sea before she left her husband?', 'sr01a_en.txt',
         "For I have married a house-cruffender, and I think he's a nice young man,she had not been that sea three weeks, I'm sure it was not for, until she began to leave the house-cruffender, you never see a anymore,"),
        ('What weather element was covering the hills in this winter morning?', 'sr08b_en.txt',
         "Like a winter's morning when the hills are glad with snow."), (
        'Did the dark night leave the boat six years ago or was it someone else?', 'sr08b_en.txt',
        'For my dark night can all her. For my dark night can all her though may he live or die. My every hope is based on him. To love will wait, to love will win. She said while tears summarized it fall. To smile dark night can all her. To smile dark night can all her. Approving my dumb fall. His cold black eyes and curly hair. His flattering tongue, my heart and snare. Gentle was he, no rake like you. To advise our maiden. To advise our maiden to slide the jacket blue. It is six long years since he left our boat.'),
        ('Describe the woman who was nameless and singingly.', 'sr08b_en.txt',
         'I know this is true but told me by bitty. Her self who is nameless is singingly. It was a lovely young lady fair. Was walking out to take the air.'),
        ('What accent or brogue did the woman laugh and joke in?', 'sr08b_en.txt',
         "There were four miles around. Her prices were transparent. She stood her own ground. She laughed and she'd joke in a rich Irish bro. She cheered up her collar this bit of a robe."),
        ('How long until lord love of his sin return to Nancy?', 'sr08a_en.txt',
         'When will you be back Lord love of His sin, or when will you be back said she, in a year or two or three at most, I return to my fair Nancy,'),
        ('What color was the horse that the lord rode while returning to Nancy?', 'sr08a_en.txt',
         'So he rode and he rode on his milk white horse, till he came to London town,'), (
        "What caused the Lord's death after discovering Nancy's death?", 'sr08a_en.txt',
        'Lady Nancy, she died as I might today. Lord love of He died too, Morrow. Lady Nancy, she died out of court court. Reward love of He died for sorrow. Lord love of He died of sorrow.'),
        ('Where were the lord and Nancy laid to rest?', 'sr08a_en.txt',
         'Lady Nancy was laid, laid in the church. Lord love of He was laid, set by her side,'), (
        'What are storms of the sea so often termed as?', 'sr19a_en.txt',
        'and why are storms upon the sea so often termed as foes?')
    ]

    results = []
    for query, expected_file, expected_content in test_queries:
        try:
            logger.info("\n--- Search Parameters ---")
            logger.info(f"Query: {query}")
            logger.info(f"Expected file: {expected_file}")
            logger.info(f"Expected content: {expected_content}")

            try:
                embedding_vector = retriever.embeddings.embed_query(query)
                logger.info(f"Generated query embedding shape: {len(embedding_vector)}")
            except Exception as e:
                logger.error(f"Error generating query embedding: {e}")
                continue

            try:
                search_results = retriever.search_vector_store(query, top_k=50)
                logger.info(f"Search completed. Found {len(search_results)} results")

                # Check for content match
                content_match = any(expected_content.lower() in doc.page_content.lower()
                                    for doc in search_results)

                # Check for document match
                doc_match = False
                correct_doc_rank = None
                for i, doc in enumerate(search_results, 1):
                    original_filename = doc.metadata.get('original_filename', '')
                    if original_filename == expected_file:
                        doc_match = True
                        correct_doc_rank = i
                        break

                results.append((query, len(search_results), content_match, doc_match))

                # Log detailed results
                logger.info("\nSearch Results Analysis:")
                logger.info(f"Content found: {content_match}")
                logger.info(f"Correct document found: {doc_match}")
                if correct_doc_rank:
                    logger.info(f"Correct document rank: {correct_doc_rank}")

                for i, doc in enumerate(search_results[:3], 1):  # Log first 3 results
                    logger.info(f"\nResult {i}:")
                    logger.info(f"File: {doc.metadata.get('original_filename', 'Unknown')}")
                    logger.info(f"Content preview: {doc.page_content[:200]}...")
                    contains_expected = expected_content.lower() in doc.page_content.lower()
                    is_correct_doc = doc.metadata.get('original_filename', '') == expected_file
                    logger.info(f"Contains expected content: {contains_expected}")
                    logger.info(f"Is correct document: {is_correct_doc}")

            except Exception as e:
                logger.error(f"Error during similarity search: {e}")
                results.append((query, 0, False, False))

        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            results.append((query, 0, False, False))

    # Calculate and log summary statistics
    total_queries = len(results)
    content_matches = sum(1 for _, _, content_match, _ in results if content_match)
    doc_matches = sum(1 for _, _, _, doc_match in results if doc_match)

    logger.info("\n=== Summary Statistics ===")
    logger.info(f"Total queries: {total_queries}")
    logger.info(f"Content matches: {content_matches} ({content_matches / total_queries * 100:.1f}%)")
    logger.info(f"Document matches: {doc_matches} ({doc_matches / total_queries * 100:.1f}%)")
    logger.info(
        f"Perfect matches (both): {sum(1 for _, _, c, d in results if c and d)} ({sum(1 for _, _, c, d in results if c and d) / total_queries * 100:.1f}%)")

    return results


def verify_dataset_integrity(dataset_path: str) -> bool:
    """Verify the integrity of the dataset."""
    try:
        logger.info(f"\nVerifying dataset integrity at: {dataset_path}")

        # Check if dataset exists
        if not os.path.exists(dataset_path):
            logger.error("Dataset path does not exist")
            return False

        # Load dataset
        ds = deeplake.load(dataset_path)

        # Check required tensors
        required_tensors = ['embedding', 'text', 'metadata']
        for tensor in required_tensors:
            if tensor not in ds.tensors:
                logger.error(f"Missing required tensor: {tensor}")
                return False
            if len(ds.tensors[tensor]) == 0:
                logger.error(f"Tensor {tensor} is empty")
                return False

        # Check consistency
        sizes = [len(ds.tensors[tensor]) for tensor in required_tensors]
        if len(set(sizes)) != 1:
            logger.error(f"Inconsistent tensor sizes: {dict(zip(required_tensors, sizes))}")
            return False

        # Sample check
        try:
            sample_idx = 0
            embedding = ds.embedding[sample_idx].numpy()
            text = ds.text[sample_idx].numpy()
            metadata = ds.metadata[sample_idx].numpy()

            logger.info("\nSample check results:")
            logger.info(f"Embedding shape: {embedding.shape}")
            logger.info(f"Text sample: {text[:100]}...")
            logger.info(f"Metadata sample: {metadata}")

            return True

        except Exception as e:
            logger.error(f"Error during sample check: {e}")
            return False

    except Exception as e:
        logger.error(f"Error verifying dataset: {e}")
        return False


def test_vectorstore_contents():
    """Test the vectorstore contents and functionality with enhanced result reporting."""
    logger.info("\nStarting vectorstore content test...")

    dataset_path = os.path.join(DATA_DIR, 'deeplake_dataset_chunk_1000')
    test_results = {
        "dataset_info": None,
        "query_results": [],
        "summary_metrics": {}
    }

    try:
        # Verify dataset integrity
        if not verify_dataset_integrity(dataset_path):
            error_msg = "Dataset integrity check failed"
            logger.error(error_msg)
            return False, {"error": error_msg}

        # Inspect dataset
        logger.info("Inspecting dataset...")
        dataset_info = inspect_dataset(dataset_path)
        test_results["dataset_info"] = dataset_info

        if "error" in dataset_info:
            logger.error(f"Dataset inspection error: {dataset_info['error']}")
            return False, test_results

        total_samples = dataset_info.get("total_samples", 0)
        logger.info(f"\nTotal samples in dataset: {total_samples}")

        if total_samples == 0:
            error_msg = "Dataset is empty"
            logger.error(error_msg)
            return False, {"error": error_msg}

        # Initialize retriever
        logger.info("\nInitializing retriever...")
        retriever = RAGRetriever(
            dataset_path=dataset_path,
            model_name='instructor',
            logger=logger
        )

        # Load existing vectorstore in read-only mode
        logger.info("Loading existing vectorstore in read-only mode...")
        retriever.vectorstore = DeepLake(
            dataset_path=dataset_path,
            embedding_function=retriever.embeddings,
            read_only=True
        )

        # Run test queries
        logger.info("\nExecuting test queries...")
        query_results = test_sample_queries(retriever)

        # Process results
        test_results["query_results"] = [
            {
                "query": query,
                "num_results": num_results,
                "content_match": content_match,
                "document_match": doc_match
            }
            for query, num_results, content_match, doc_match in query_results
        ]

        # Calculate summary metrics
        total_queries = len(query_results)
        content_matches = sum(1 for _, _, content_match, _ in query_results if content_match)
        doc_matches = sum(1 for _, _, _, doc_match in query_results if doc_match)
        perfect_matches = sum(1 for _, _, c, d in query_results if c and d)

        test_results["summary_metrics"] = {
            "total_queries": total_queries,
            "content_matches": content_matches,
            "content_match_rate": (content_matches / total_queries) * 100 if total_queries > 0 else 0,
            "document_matches": doc_matches,
            "document_match_rate": (doc_matches / total_queries) * 100 if total_queries > 0 else 0,
            "perfect_matches": perfect_matches,
            "perfect_match_rate": (perfect_matches / total_queries) * 100 if total_queries > 0 else 0
        }

        # Log results
        log_test_results("vectorstore_content_test", test_results)

        logger.info("\nTest Results Summary:")
        logger.info(f"Total queries: {total_queries}")
        logger.info(
            f"Content matches: {content_matches} ({test_results['summary_metrics']['content_match_rate']:.1f}%)")
        logger.info(f"Document matches: {doc_matches} ({test_results['summary_metrics']['document_match_rate']:.1f}%)")
        logger.info(
            f"Perfect matches: {perfect_matches} ({test_results['summary_metrics']['perfect_match_rate']:.1f}%)")

        return True, test_results

    except Exception as e:
        logger.error(f"Error in test execution: {e}", exc_info=True)
        test_results["error"] = str(e)
        return False, test_results

    except Exception as e:
        logger.error(f"Error in test execution: {e}", exc_info=True)
        test_results["error"] = str(e)
        return False, test_results


if __name__ == "__main__":
    logger.info("Starting vectorstore tests...")
    logger.info("=" * 80)

    success, results = test_vectorstore_contents()

    if success:
        logger.info("\nVectorstore content test completed successfully.")

        if "dataset_info" in results:
            logger.info(f"Dataset size: {results['dataset_info'].get('total_samples', 0)} samples")

        if "query_results" in results:
            logger.info("\nDetailed Query Results:")
            for result in results["query_results"]:
                logger.info("\n" + "=" * 50)
                logger.info(f"Query: {result['query']}")
                logger.info(f"Results found: {result['num_results']}")
                logger.info(f"Content match: {'Yes' if result['content_match'] else 'No'}")
                logger.info(f"Document match: {'Yes' if result['document_match'] else 'No'}")

                # Add detailed result information if available
                if 'top_results' in result:
                    logger.info("\nTop 3 Results:")
                    for i, doc_info in enumerate(result['top_results'][:3], 1):
                        logger.info(f"\nResult {i}:")
                        logger.info(f"Filename: {doc_info.get('filename', 'Unknown')}")
                        logger.info(f"Content preview: {doc_info.get('content_preview', '')[:200]}...")
                        logger.info("Metadata:")
                        for key, value in doc_info.get('metadata', {}).items():
                            logger.info(f"  {key}: {value}")
                        logger.info(f"Contains expected content: {doc_info.get('contains_expected', False)}")
                        logger.info(f"Is correct document: {doc_info.get('is_correct_doc', False)}")

        if "summary_metrics" in results:
            logger.info("\nOverall Performance:")
            logger.info(f"Total queries: {results['summary_metrics']['total_queries']}")
            logger.info(f"Content match rate: {results['summary_metrics']['content_match_rate']:.1f}%")
            logger.info(f"Document match rate: {results['summary_metrics']['document_match_rate']:.1f}%")
            logger.info(f"Perfect match rate: {results['summary_metrics']['perfect_match_rate']:.1f}%")
    else:
        logger.error("\nVectorstore content test failed.")
        if "error" in results:
            logger.error(f"Error message: {results['error']}")

    logger.info("\nTest execution completed. Check the logs directory for detailed results.")