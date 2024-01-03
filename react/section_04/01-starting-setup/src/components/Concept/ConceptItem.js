import Image from "../UI/Image";
import "./ConceptItem.css";

function ConceptItem(props) {
  return (
    <li className="concept">
      <Image image={props.image} name={props.title} />
      <h2>{props.title}</h2>
      <p>{props.description}</p>
    </li>
  );
}

export default ConceptItem;
