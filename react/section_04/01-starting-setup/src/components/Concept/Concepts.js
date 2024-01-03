import ConceptItem from "./ConceptItem";
import "./Concepts.css";

function Concepts(props) {
  return (
    <ul className="concepts">
      <ConceptItem
        title={props.concepts[0].title}
        image={props.concepts[0].image}
        description={props.concepts[0].description}
      />
      <ConceptItem
        title={props.concepts[1].title}
        image={props.concepts[1].image}
        description={props.concepts[1].description}
      />
      <ConceptItem
        title={props.concepts[2].title}
        image={props.concepts[2].image}
        description={props.concepts[2].description}
      />
    </ul>
  );
}

export default Concepts;
